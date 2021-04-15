import torch
import torch.nn as nn
import antialiased_cnns
from combinet.src.architecture.building_blocks import SeparableConv2d, _Upsample

class BasicLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, args):
        super().__init__()
        self.args = args
        self.bn = nn.BatchNorm2d(in_channels, track_running_stats=False)
        self.relu = nn.ReLU(True)
        self.conv = SeparableConv2d(in_channels, growth_rate,
                                kernel_size=3, stride=1, padding=1, bias=True, args=args)
        self.dropout = nn.Dropout2d(0.05, True)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        return x


class Dense(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False,args=None):
        super().__init__()
        self.args = args
        self.upsample = upsample
        self.layers = nn.ModuleList([BasicLayer(
            in_channels + i*growth_rate, growth_rate, args=args)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features, 1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1) 
            return x

class Downsample(nn.Module):
    def __init__(self, in_channels, args=None):
        super().__init__()
        self.args = args
        self.bn = nn.BatchNorm2d(in_channels, track_running_stats=False)
        self.relu = nn.ReLU(True)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1,stride=1, padding=0, bias=True)
        self.dropout = nn.Dropout2d(0.05, True)
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1), 
                                  antialiased_cnns.BlurPool(in_channels, stride=2, filt_size=2))
    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, args=None):
        super().__init__()
        self.args = args
        self.project = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=True)
        self.interpolate = _Upsample(mode='bilinear', align_corners=False)

    def forward(self, x, skip):
        output_shape = list(x.shape[-2:])
        output_shape[0] = max(output_shape[0]*2, skip.shape[2])
        output_shape[1] = max(output_shape[1]*2, skip.shape[3])
        out =  self.interpolate(x, size=output_shape)
        out = self.project(out)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, args=None):
        super().__init__()
        self.args = args
        self.bottleneck= Dense(
                in_channels, growth_rate, n_layers, upsample=True, args=args)

    def forward(self, x):
        return self.bottleneck(x)

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]
