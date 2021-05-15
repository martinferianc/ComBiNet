import torch.nn as nn
from combinet.src.architecture.layers import *
from combinet.src.architecture.aspp import ASPP

def CombiNet51(args, n_classes):
    return CombiNet(
        in_channels=3, down_blocks=(4, 4, 4, 4, 4),
        up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4, aspps = [False, True, True, True, False],
        growth_rate=12, out_chans_first_conv=48, n_classes=n_classes,  args=args)


def CombiNet62(args, n_classes):
    return CombiNet(
        in_channels=3, down_blocks=(5, 5, 5, 5, 5),
        up_blocks=(5, 5, 5, 5, 5), bottleneck_layers=5, aspps = [False, True, True, True, False],
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes,  args=args)


def CombiNet87(args, n_classes):
    return CombiNet(
        in_channels=3, down_blocks=(5, 6, 7, 8, 9),
        up_blocks=(9, 8, 7, 6, 5), bottleneck_layers=10, aspps = [False, True, True, True, False],
        growth_rate=16, out_chans_first_conv=48, n_classes=n_classes,  args=args)

class CombiNet(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(4, 4, 4, 4, 4),
                 up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4, aspps = [True, True, True, True, True],
                 growth_rate=12, out_chans_first_conv=48, n_classes=12, args=None):
        """
        You will notice that the REPEAT BLOCK is not explicitly implemented as a separate module because
        it makes managing proper caching of the activations and skip-connections too difficult and the 
        memory consumption rises significantly
        Hence you will notice that the each part of the repeat block is implemented SEPARATELY and we 
        let PyTorch handle proper memory management, which is still far from perfect, any help is welcome! 
        """
        super().__init__()
        self.down_blocks = down_blocks
        self.up_blocks = up_blocks
        self.args = args

        self.asppBlocks = nn.ModuleList([])
        self.denseDown = nn.ModuleList([])
        self.downsamples = nn.ModuleList([])

        self.firstconv = nn.Conv2d(in_channels=in_channels,
                                               out_channels=out_chans_first_conv, kernel_size=3,
                                               stride=1, padding=1, bias=True)
        cur_channels_count = out_chans_first_conv
        skip_connection_channel_counts = []
        output_stride = 1

        self.dummy_paramater = nn.Parameter(
            torch.empty(1))

        for i in range(len(down_blocks)):
            self.denseDown.append(
                Dense(cur_channels_count, growth_rate, down_blocks[i], args=args))
            cur_channels_count += (growth_rate*down_blocks[i])
            skip_connection_channel_counts.insert(0, cur_channels_count)
            if aspps[i]:
                self.asppBlocks.insert(0, ASPP(cur_channels_count, cur_channels_count, output_stride, args=args))
            else:
                self.asppBlocks.insert(0, nn.Identity())
            self.downsamples.append(Downsample(cur_channels_count, args=args))
            output_stride*=2


        self.bottleneck= Bottleneck(cur_channels_count,
                                                 growth_rate, bottleneck_layers, args=args)
        prev_block_channels = growth_rate*bottleneck_layers

        self.upsamples = nn.ModuleList([])
        self.denseUp = nn.ModuleList([])
        for i in range(len(up_blocks)-1):
            self.upsamples.append(Upsample(
                prev_block_channels, prev_block_channels, args=args))
            cur_channels_count = prev_block_channels + \
                skip_connection_channel_counts[i]

            self.denseUp.append(Dense(
                cur_channels_count, growth_rate, up_blocks[i],
                upsample=True, args=args))
            prev_block_channels = growth_rate*up_blocks[i]

        self.upsamples.append(Upsample(
            prev_block_channels, prev_block_channels, args=args))
        cur_channels_count = prev_block_channels + \
            skip_connection_channel_counts[-1]

        self.denseUp.append(Dense(
            cur_channels_count, growth_rate, up_blocks[-1],
            upsample=False, args=args))
        cur_channels_count += growth_rate*up_blocks[-1]


        self.finalConv = nn.Conv2d(in_channels=cur_channels_count,
                                   out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight,  nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d) and m.bias is not None:
                m.bias.data.uniform_(0,1)
 
    def eval(self, mode="wa"):
        super().eval()
        def _apply_dropout(m):
            if type(m) == nn.Dropout2d:
                m.train()
        if mode == "train":
            self.apply(_apply_dropout)

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseDown[i](out)
            skip_connections.append(out)
            out = self.downsamples[i](out)
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            skip = skip_connections.pop()
            skip = self.asppBlocks[i](skip)
            out = self.upsamples[i](out, skip)
            out = self.denseUp[i](out)
        out = self.finalConv(out)
        out = self.softmax(out)
        return out
