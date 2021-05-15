import numpy as np
from prettytable import PrettyTable
import logging
import torch.nn as nn
from combinet.src.architecture.building_blocks import _Upsample

import torch 
from functools import partial
import antialiased_cnns

def count_parameters_macs(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    logging.info("### Parameters in a model ###")
    logging.info(table)
    logging.info("#### Total trainable params: {} ####".format(total_params))
    custom_modules_mapping = {
        nn.Dropout2d: dropout_macs_counter_hook,
        antialiased_cnns.BlurPool: blurpool_macs_counter_hook,
        _Upsample: upsample_macs_counter_hook
    }
    logging.write = logging.info
    mac = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True, custom_modules_hooks=custom_modules_mapping)
    logging.info("#### Total MACs: {} ####".format(mac))


def get_model_complexity_info(model, input_res,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None,
                              verbose=False, ignore_modules=[],
                              custom_modules_hooks={}):
    assert type(input_res) is tuple
    assert len(input_res) >= 1
    assert isinstance(model, nn.Module)
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    macs_model = add_macs_counting_methods(model)
    macs_model.eval()
    macs_model.start_macs_count(verbose=verbose,
                                  ignore_list=ignore_modules)
    if input_constructor:
        input = input_constructor(input_res)
        _ = macs_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty((1, *input_res),
                                             dtype=next(macs_model.parameters()).dtype,
                                             device=next(macs_model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

        _ = macs_model(batch)

    macs_count = macs_model.compute_average_macs_cost()
    if print_per_layer_stat:
        print_model_with_macs(macs_model, macs_count)
    macs_model.stop_macs_count()
    CUSTOM_MODULES_MAPPING = {}

    if as_strings:
        return macs_to_string(macs_count)

    return macs_count


def macs_to_string(macs, units='GMac', precision=2):
    if units is None:
        if macs // 10**9 > 0:
            return str(round(macs / 10.**9, precision)) + ' GMac'
        elif macs // 10**6 > 0:
            return str(round(macs / 10.**6, precision)) + ' MMac'
        elif macs // 10**3 > 0:
            return str(round(macs / 10.**3, precision)) + ' KMac'
        else:
            return str(macs) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(macs / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(macs / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(macs / 10.**3, precision)) + ' ' + units
        else:
            return str(macs) + ' Mac'


def accumulate_macs(self):
    if is_supported_instance(self):
        return self.__macs__
    return sum(m.accumulate_macs() for m in self.children())


def print_model_with_macs(model, total_macs, units='GMac',
                           precision=3):

    
    def macs_repr(self):
        accumulated_macs_cost = self.accumulate_macs() / model.__batch_counter__
        return ', '.join([macs_to_string(accumulated_macs_cost,
                                          units=units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_macs_cost / total_macs),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_macs = accumulate_macs.__get__(m)
        macs_extra_repr = macs_repr.__get__(m)
        if m.extra_repr != macs_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = macs_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_macs'):
            del m.accumulate_macs

    model.apply(add_extra_repr)
    logging.info(repr(model))
    model.apply(del_extra_repr)



def add_macs_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_macs_count = start_macs_count.__get__(net_main_module)
    net_main_module.stop_macs_count = stop_macs_count.__get__(net_main_module)
    net_main_module.reset_macs_count = reset_macs_count.__get__(net_main_module)
    net_main_module.compute_average_macs_cost = compute_average_macs_cost.__get__(
                                                    net_main_module)

    net_main_module.reset_macs_count()

    return net_main_module


def compute_average_macs_cost(self):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.

    Returns current mean macs consumption per image.

    """

    for m in self.modules():
        m.accumulate_macs = accumulate_macs.__get__(m)

    macs_sum = self.accumulate_macs()

    for m in self.modules():
        if hasattr(m, 'accumulate_macs'):
            del m.accumulate_macs

    return macs_sum / self.__batch_counter__


def start_macs_count(self, **kwargs):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.

    Activates the computation of mean macs consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_macs_counter_hook_function(module, verbose, ignore_list):
        if type(module) in ignore_list:
            pass
        elif is_supported_instance(module):
            if hasattr(module, '__macs_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                handle = module.register_forward_hook(
                                        CUSTOM_MODULES_MAPPING[type(module)])
            else:
                handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
            module.__macs_handle__ = handle
        else:
            if (
                verbose
                and type(module) not in (nn.Sequential, nn.ModuleList)
                and type(module) not in seen_types
            ):
                logging.info('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.')

        seen_types.add(type(module))

    self.apply(partial(add_macs_counter_hook_function, **kwargs))


def stop_macs_count(self):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.

    Stops computing the mean macs consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_macs_counter_hook_function)


def reset_macs_count(self):
    """
    A method that will be available after add_macs_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_macs_counter_variable_or_reset)


# ---- Internal functions
def empty_macs_counter_hook(module, input, output):
    module.__macs__ += 0


def dropout_macs_counter_hook(module, input, output):
    input = input[0]
    module.__macs__ += int(np.prod(input.shape))


def blurpool_macs_counter_hook(module, input, output):
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = [module.filt_size, module.filt_size]
    in_channels = module.channels
    out_channels = module.channels
    groups = module.channels

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_macs = conv_per_position_macs * active_elements_count

    module.__macs__ += int(overall_macs)

def upsample_macs_counter_hook(module, input, output):
    module.__macs__ += int(np.prod(output.shape))

def relu_macs_counter_hook(module, input, output):
    module.__macs__ += int(np.prod(output.shape))

def pool_macs_counter_hook(module, input, output):
    module.__macs__ += int(np.prod(input[0].shape))

def bn_macs_counter_hook(module, input, output):
    input = input[0]

    batch_macs = np.prod(input.shape)
    if module.affine:
        batch_macs *= 2
    if module.track_running_stats is False: 
        batch_macs *= 2
    module.__macs__ += int(batch_macs)


def conv_macs_counter_hook(conv_module, input, output):
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_macs = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_macs = conv_per_position_macs * active_elements_count

    bias_macs = 0

    if conv_module.bias is not None:

        bias_macs = out_channels * active_elements_count

    overall_macs = overall_conv_macs + bias_macs

    conv_module.__macs__ += int(overall_macs)


def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size



def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_macs_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__macs__'):
            print('Warning: variables __macs__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' ptmacs can affect your code!')
        module.__macs__ = 0


CUSTOM_MODULES_MAPPING = {}

MODULES_MAPPING = {
    # convolutions
    nn.Conv2d: conv_macs_counter_hook,
    # activations
    nn.ReLU: relu_macs_counter_hook,
    # poolings
    nn.MaxPool2d: pool_macs_counter_hook,
    nn.AdaptiveAvgPool2d: pool_macs_counter_hook,
    # BNs
    nn.BatchNorm2d: bn_macs_counter_hook,
    # Upscale
    nn.Upsample: upsample_macs_counter_hook,
}


def is_supported_instance(module):
    return (
        type(module) in MODULES_MAPPING
        or type(module) in CUSTOM_MODULES_MAPPING
    )


def remove_macs_counter_hook_function(module):
    if is_supported_instance(module) and hasattr(module, '__macs_handle__'):
        module.__macs_handle__.remove()
        del module.__macs_handle__