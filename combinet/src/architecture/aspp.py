import torch
import torch.nn as nn
from combinet.src.architecture.building_blocks import SeparableConv2d, _Upsample

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, args):
        super(_ASPPModule, self).__init__()
        self.args = args
        if kernel_size!=1:
            self.conv = SeparableConv2d(inplanes, planes, kernel_size = kernel_size,
                                    stride = 1, padding = padding, dilation = dilation, bias = False, args=args)
        else:
            self.conv = nn.Conv2d(inplanes, planes, kernel_size = kernel_size,
                                    stride = 1, padding = padding, dilation = dilation, bias = False)
        self.bn = nn.BatchNorm2d(inplanes, track_running_stats=False)
        self.relu =  nn.ReLU(True)
        
    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes, output_stride, args):
        super(ASPP, self).__init__()
        self.args = args
        self.dilations = None
        if output_stride == 1:
            self.dilations = [1, 24, 40, 56]
        elif output_stride == 2:
            self.dilations = [1, 12, 20, 28]
        elif output_stride == 4:
            self.dilations = [1, 5, 10, 14]
        elif output_stride == 8:
            self.dilations = [1, 3, 5, 7]
        elif output_stride == 16:
            self.dilations = [1, 2, 3, 4]
        else:
            raise NotImplementedError
    
        self.output_stride = output_stride

        self.aspp1 = _ASPPModule(
            inplanes, 32, 1, padding=0, dilation=self.dilations[0],args=args)
        self.aspp2 = _ASPPModule(
            inplanes, 32, 3, padding=self.dilations[1], dilation=self.dilations[1],args=args)
        self.aspp3 = _ASPPModule(
            inplanes, 32, 3, padding=self.dilations[2], dilation=self.dilations[2],args=args)
        self.aspp4 = _ASPPModule(
            inplanes, 32, 3, padding=self.dilations[3], dilation=self.dilations[3],args=args)

        self.global_avg_pool = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(
                                                 inplanes, 32, 1, stride=1, bias=True),
                                             nn.ReLU(True)])
        self.conv = nn.Conv2d(160, outplanes, 1, bias=False)
        self.bn = nn.BatchNorm2d(outplanes, track_running_stats=False)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout2d(0.05)
        self.interpolate = _Upsample(mode='bilinear', align_corners=True)


    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = x
        for layer in self.global_avg_pool:
            x5 = layer(x5)
        x5 = self.interpolate(x5, size=x4.size()[
                           2:])
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

    def extra_repr(self):
        return 'dilations={}, output_stride={}'.format(self.dilations, self.output_stride)
