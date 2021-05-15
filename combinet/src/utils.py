import os
import numpy as np
import torch
import shutil
import random
import pickle
import sys
import time
import glob
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import shutil

def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def save_model(model, args, special_info=""):
  torch.save(model.state_dict(), os.path.join(args.save, 'weights'+special_info+'.pt'))

  with open(os.path.join(args.save, 'args.pt'), 'wb') as handle:
    pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_entropy_output(output_mean):
    return -torch.sum(torch.log(output_mean+1e-8)*output_mean, dim=1)


def save_pickle(data, path, overwrite=False):
  path = check_path(path) if not overwrite else path
  with open(path, 'wb') as fp:
      pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    file = open(path, 'rb')
    return pickle.load(file)

def load_model(model, model_path):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k,v in state_dict.items()}
    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
def create_exp_dir(path, scripts_to_save=None):
  path = check_path(path)
  os.mkdir(path)

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def check_path(path):
  if os.path.exists(path):
    filename, file_extension = os.path.splitext(path)
    counter = 0
    while os.path.exists(filename+"_"+str(counter)+file_extension):
      counter+=1
    return filename+"_"+str(counter)+file_extension
  return path

def model_to_gpus(model, args):
  if args.gpu!= -1:
    device = torch.device("cuda:"+str(args.gpu))
    model = model.to(device)
  return model

def parse_args(args, label=""):
    dataset = args.dataset if hasattr(args, 'dataset') else ""
    if args.load=="EXP":
      new_path = '{}-{}-{}'.format(dataset,label, time.strftime("%Y%m%d-%H%M%S"))
    else:
      new_path = 'load-{}-{}-{}'.format(dataset,label, time.strftime("%Y%m%d-%H%M%S"))

    create_exp_dir(
      new_path, scripts_to_save=glob.glob('*.py') + \
                            glob.glob('../*.py') + \
                            glob.glob('../../src/**/*.py', recursive=True))
    args.save = new_path
    if args.load!="EXP":
      loading_path = os.path.split(args.load)
      filename = loading_path[-1]
      if not os.path.isdir(args.load):
        shutil.copy(args.load, os.path.join(new_path, filename))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    log_path = os.path.join(args.save, 'log.log')
    log_path = check_path(log_path)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    print('Experiment dir : {}'.format(args.save))

    writer = SummaryWriter(
        log_dir=args.save+"/",max_queue=5)

    if torch.cuda.is_available() and args.gpu!=-1:
      logging.info('## GPUs available = {} ##'.format(args.gpu))
      torch.cuda.set_device(args.gpu)
      cudnn.benchmark = True
      cudnn.enabled = True
      torch.cuda.manual_seed(args.seed)
    else:
      logging.info('## No GPUs detected ##')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    logging.info("## Args = %s ##", args)

    path = os.path.join(args.save, 'results.pickle')
    path= check_path(path)
    results = {
        "dataset": args.dataset if hasattr(args, 'dataset') else "",
        "model": args.model if hasattr(args, 'model') else "",
        "global_error": {},
        "class_error": {},
        "nll": {},
        "miou": {},
        "ece": {},
        "entropy": {},
        "mean_dice": {},
    }

    save_pickle(results, path, True)

    return args, writer

      

          
  
