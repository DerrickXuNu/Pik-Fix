"""
An implementation combining dense121, unet and residual dense block. Reference image added
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34
from torchvision.models.resnet import BasicBlock, Bottleneck

from models.networks import _DenseBlock, _Transition, RDB, GaussianHistogram, AttentionExtractModule
from models.warpnet import WarpNet


class HistogramLayerLocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_layer = GaussianHistogram(bins=256, min=-1., max=1., sigma=0.01, require_grad=False)

    def forward(self, x, ref, attention_mask=None):
        channels = ref.shape[1]
        if len(x.shape) == 3:
            ref = F.interpolate(ref,
                                size=(x.shape[1], x.shape[2]),
                                mode='bicubic')
            if not type(attention_mask) == type(None):
                attention_mask = torch.unsqueeze(attention_mask, 1)
                attention_mask = F.interpolate(attention_mask,
                                               size=(x.shape[1], x.shape[2]),
                                               mode='bicubic')
        else:
            ref = F.interpolate(ref,
                                size=(x.shape[2], x.shape[3]),
                                mode='bicubic')
            if not type(attention_mask) == type(None):
                attention_mask = torch.unsqueeze(attention_mask, 1)
                attention_mask = F.interpolate(attention_mask,
                                               size=(x.shape[2], x.shape[3]),
                                               mode='bicubic')
                attention_mask = torch.flatten(attention_mask, start_dim=1, end_dim=-1)

        layers = []
        for i in range(channels):
            input_channel = torch.flatten(ref[:, i, :, :], start_dim=1, end_dim=-1)
            input_hist, hist_dist = self.hist_layer(input_channel, attention_mask)
            hist_dist = hist_dist.view(-1, 256, ref.shape[2], ref.shape[3])
            layers.append(hist_dist)

        return torch.cat(layers, 1)


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


class HistFusionModule(nn.Module):
    """
    Global pooling fused with histogram
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.RDB = RDB(out_features, 4, 32)

    def forward(self, feature):
        feature = self.conv(feature)
        feature = self.RDB(feature)

        return feature


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, current_channels, prev_channels, out_channels,
                 bilinear=True, nDenseLayer=3, growthRate=32, global_pool=False):
        super().__init__()
        self.global_pool = global_pool
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        else:
            self.up = nn.ConvTranspose2d(current_channels, current_channels, kernel_size=2, stride=2)

        self.RDB = RDB(current_channels, nDenseLayer, growthRate)
        self.conv = DoubleConv(current_channels + prev_channels, out_channels)

    def forward(self, x1, x2):
        h, w = x2.shape[2], x2.shape[3]
        if not self.global_pool:
            x1 = self.up(x1)
            # input is CHW
            diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
            diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
            # in case input size are odd
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        else:
            x1 = F.upsample(x1, size=(h, w), mode='bilinear')

        x1 = self.RDB(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Dense121UnetHistogramAttention(nn.Module):
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

    def __init__(self, args, color_pretrain=False, growth_rate=32, block_config=(6, 12, 24, 48),
                 num_init_features=64, bn_size=4):
        super(Dense121UnetHistogramAttention, self).__init__()
        self.color_pretrain = color_pretrain

        # reference local histogram layer
        self.hist_layer_local = HistogramLayerLocal()

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

            # downsampling
            trans = _Transition(num_input_features=num_features,
                                num_output_features=num_features // 2)
            self.features.add_module('transition%d' % (i + 1), trans)
            num_features = num_features // 2

        # histogram distribution fusion part, feature + similarity mask + histogram
        self.hf_1 = HistFusionModule(128 + 1 + 256 * 2, 128)
        self.hf_2 = HistFusionModule(256 + 1 + 256 * 2, 256)
        self.hf_3 = HistFusionModule(512 + 1 + 256 * 2, 512)
        self.hf_4 = HistFusionModule(1024 + 1 + 256 * 2, 1024)

        # Decoder Part
        self.up0 = Up(1024, 2048, 1024, args['bilinear'], args['nDenseLayer'][0], args['growthRate'])
        self.up1 = Up(1024, 1024, 512, args['bilinear'], args['nDenseLayer'][0], args['growthRate'])
        self.up2 = Up(512, 512, 256, args['bilinear'], args['nDenseLayer'][1], args['growthRate'])
        self.up3 = Up(256, 256, 128, args['bilinear'], args['nDenseLayer'][2], args['growthRate'])
        self.up4 = Up(128, 64, 64, args['bilinear'], args['nDenseLayer'][3], args['growthRate'])

        nChannels = args['input_channel']
        self.conv_final = nn.Conv2d(64, nChannels, kernel_size=3, padding=1, bias=True)
        self.warp_net = WarpNet()

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

    def normalize_data(self, x):
        """
        Normalize the data for attention module
        """
        device = x.device

        mean_r = torch.ones(1, x.shape[2], x.shape[3]) * 0.485
        mean_g = torch.ones(1, x.shape[2], x.shape[3]) * 0.456
        mean_b = torch.ones(1, x.shape[2], x.shape[3]) * 0.406
        mean = torch.cat((mean_r, mean_g, mean_b), dim=0)
        mean = mean.to(device)

        std_r = torch.ones(1, x.shape[2], x.shape[3]) * 0.229
        std_g = torch.ones(1, x.shape[2], x.shape[3]) * 0.224
        std_b = torch.ones(1, x.shape[2], x.shape[3]) * 0.225
        std = torch.cat((std_r, std_g, std_b), dim=0)
        std = std.to(device)

        normalized_data = (x - mean) / std
        return normalized_data

    def forward(self, x, x_gray, ref, ref_gray, att_model):
        """
        :param x: input data
        :param gt: gt_data
        :param gt: gt_gray
        :param att_model: pretrained resent34
        """
        # shallow conv
        feature0 = self.features.relu0(self.features.conv0_0(x))
        down0 = self.features.pool0(feature0)

        # normalize data for attention mask
        normalized_ref = self.normalize_data(ref_gray.repeat(1, 3, 1, 1))
        normalized_x = self.normalize_data(x_gray.repeat(1, 3, 1, 1))

        # attention mask for both input and ground truth(size divide 4, 8, 16, 32)
        ref_attention_masks, ref_res_features = att_model(normalized_ref)
        x_attention_masks, x_res_features = att_model(normalized_x)

        # generate histogram for different size
        ref_resize_by_8 = F.avg_pool2d(ref, 8)
        x_resize_by_8 = F.avg_pool2d(x, 8)
        ref_hist = self.hist_layer_local(x_resize_by_8, ref_resize_by_8)

        # generate the similarity map and wrapped features
        sim_feature = self.warp_net(ref_hist,
                                    x_res_features[0], x_res_features[1], x_res_features[2], x_res_features[3],
                                    ref_res_features[0], ref_res_features[1], ref_res_features[2], ref_res_features[3])

        # dense block 1
        feature1 = self.features.denseblock1(down0)
        down1 = self.features.transition1(feature1)
        down1 = torch.cat([down1, sim_feature[0][1], sim_feature[0][0]], 1)
        down1 = self.hf_1(down1)

        # dense block 2
        feature2 = self.features.denseblock2(down1)
        down2 = self.features.transition2(feature2)
        down2 = torch.cat([down2, sim_feature[1][1], sim_feature[1][0]], 1)
        down2 = self.hf_2(down2)

        # dense block3
        feature3 = self.features.denseblock3(down2)
        down3 = self.features.transition3(feature3)
        down3 = torch.cat([down3, sim_feature[2][1], sim_feature[2][0]], 1)
        down3 = self.hf_3(down3)

        # dense block 4
        feature4 = self.features.denseblock4(down3)
        down4 = self.features.transition4(feature4)
        down4 = torch.cat([down4, sim_feature[3][1], sim_feature[3][0]], 1)
        down4 = self.hf_4(down4)

        # up
        up = self.up0(down4, feature4)
        up = self.up1(up, feature3)
        up = self.up2(up, feature2)
        up = self.up3(up, feature1)
        up = self.up4(up, feature0)

        output = self.conv_final(up)
        results = {'output': output}
        return results


if __name__ == '__main__':
    # unit test
    data = torch.randn((2, 1, 256, 256))
    gt = torch.randn((2, 3, 256, 256))
    gt_gray = torch.randn((2, 1, 256, 256))

    data = data.cuda()
    gt = gt.cuda()
    gt_gray = gt_gray.cuda()

    base_resnet34 = resnet34(pretrained=True)
    att_model = AttentionExtractModule(BasicBlock, [3, 4, 6, 3])
    att_model.load_state_dict(base_resnet34.state_dict())
    att_model.cuda()
    att_model.eval()

    args = {'input_channel': 3, 'growthRate': 32, 'bilinear': True, 'drop_rate': 0.5,
            'nDenseLayer': [8, 12, 6, 4], 'pretrained': True}
    model = Dense121UnetHistogramAttention(args)
    model.cuda()
    output = model(data, gt, gt_gray, att_model)
