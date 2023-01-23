"""
An implementation combining dense121, unet and residual dense block
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks import sub_pixel, _DenseBlock, _Transition, RDB
from collections import OrderedDict


class DoubleConv(nn.Module):
    """
    Double convoltuion
    Args:
        in_channels: input channel num
        out_channels: output channel num
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, current_channels, prev_channels, out_channels,
                 bilinear=True, nDenseLayer=3, growthRate=32):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:
            self.up = nn.ConvTranspose2d(current_channels, current_channels, kernel_size=2, stride=2)

        self.RDB = RDB(current_channels, nDenseLayer, growthRate)
        self.conv = DoubleConv(current_channels + prev_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        # in case input size are odd
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x1 = self.RDB(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Dense121Unet(nn.Module):
    """
    A combination of Dense121, Unet and residual dense block. The style is a little wired since
    we need to comply the same name rule as the torch.hub.densenet121 to retrieve pretrained model
    Args:
        args: some optional params
        growth_rate: don't change it since we are using pretrained weights
        block_config: don't change it since we are using pretrained weights
        num_init_features: don't change it since we are using pretrained weights
        bn_size: don't change it since we are using pretrained weights
    """

    def __init__(self, args, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4):
        super(Dense121Unet, self).__init__()
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0_0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=1,
                                  padding=3, bias=False)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Encoder part
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=args['drop_rate']
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Decoder Part
        self.up1 = Up(1024, 1024, 512, args['bilinear'], args['nDenseLayer'][0], args['growthRate'])
        self.up2 = Up(512, 512, 256, args['bilinear'], args['nDenseLayer'][1], args['growthRate'])
        self.up3 = Up(256, 256, 128, args['bilinear'], args['nDenseLayer'][2], args['growthRate'])
        self.up4 = Up(128, 64, 64, args['bilinear'], args['nDenseLayer'][3], args['growthRate'])

        nChannels = args['input_channel']
        self.conv_final = nn.Conv2d(64, nChannels, kernel_size=3, padding=1, bias=True)

        if args['pretrained']:
            self.load_pretrained()

    def load_pretrained(self):
        pretrained_model = torch.hub.load('pytorch/vision:v0.4.0', 'densenet121', pretrained=True)
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.load_state_dict(model_dict)

    def forward(self, x):
        # shallow conv
        feature0 = self.features.relu0(self.features.conv0_0(x))
        down0 = self.features.pool0(feature0)

        # dense block 1
        feature1 = self.features.denseblock1(down0)
        down1 = self.features.transition1(feature1)
        # dense block 2
        feature2 = self.features.denseblock2(down1)
        down2 = self.features.transition2(feature2)
        # dense block3
        feature3 = self.features.denseblock3(down2)
        down3 = self.features.transition3(feature3)
        # dense block 4
        feature4 = self.features.denseblock4(down3)

        # up
        up = self.up1(feature4, feature3)
        up = self.up2(up, feature2)
        up = self.up3(up, feature1)
        up = self.up4(up, feature0)

        output = self.conv_final(up)
        results = {'output': output}
        return results
