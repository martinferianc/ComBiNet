import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, args=None):
        super(SeparableConv2d, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(inplanes, inplanes, (1, kernel_size), stride=(1, stride),
                        padding=(0, padding), dilation=(1,dilation), groups=inplanes, bias=True)
        self.conv2 = nn.Conv2d(inplanes, inplanes, (kernel_size, 1), stride=(stride, 1),
                            padding=(padding, 0), dilation=(dilation,1), groups=inplanes,bias=False)
        self.bn = nn.BatchNorm2d(inplanes, track_running_stats=False)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class _Upsample(nn.Upsample):
    def __init__(self,
                 mode, align_corners):
        super(_Upsample, self).__init__( mode=mode, align_corners=align_corners)
       

    def forward(self, input, size):
        self.size = size
        return F.interpolate(input, size, self.scale_factor, self.mode, self.align_corners)



