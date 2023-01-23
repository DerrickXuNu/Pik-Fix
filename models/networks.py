"""
Some common blocks that may be called several times by different models
"""
import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet


# ++++++++++++++++++++++++++++++++ For Residual Dense Neural Network ++++++++++++++++++++++++++++++++++ #
class make_dense(nn.Module):
    """
    Base implementation of residual dense block
    Args:
        nChannels: input channels
        growthRate:  output channels
        kernel size: convolution kernel size
    """

    def __init__(self, nChannels, growthRate, kernel_size=3, bn=False):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate,
                              kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        # TODO: Add drop out here
        out = torch.cat((x, out), 1)
        return out


class RDB(nn.Module):
    """
    Residual dense block
        Args:
        nChannels: input channels
        nDenselayer:  # of forwarding layer in the block
        growthRate: convolution kernel size
    """

    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels,
                                  kernel_size=1,
                                  padding=0,
                                  bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class sub_pixel(nn.Module):
    """
    Up sampling use PixelShuffle
    Args:
        scale: up scale
    """

    def __init__(self, scale, act=False):
        super(sub_pixel, self).__init__()
        modules = [nn.PixelShuffle(scale)]
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.body(x)
        return x


# ++++++++++++++++++++++++++++++++ For Pretrained Dense Neural Network ++++++++++++++++++++++++++++++++++ #
def _bn_function_factory(relu, conv):
    """
    Basic concatenation function
    :param relu: relu function
    :param conv: conv function
    :return:
    """

    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(concated_features))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    """
    Basic dense layer in denseblock
    Args:
        num_input_features: input feature channels
        growth_rate: increased feature num every layer
        bn_size: bottle neck scale
        drop_rate: dropping out probability
    """

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.relu1, self.conv1)
        bottleneck_output = bn_function(*prev_features)

        new_features = self.conv2(self.relu2(bottleneck_output))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)

        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


# ++++++++++++++++++++++++++ For PSP(Pyramid Spatial polling) Module ++++++++++++++++++++++++++++++++
class PSPModule(nn.Module):
    def __init__(self, in_features, inter_features, out_features=64, sizes=(56, 28, 14, 7)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_features, inter_features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(inter_features * len(sizes) + in_features, out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, in_features, inter_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_features, inter_features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


# ++++++++++++++++++++++++++ For Receptive Filed Block Module ++++++++++++++++++++++++++++++++
class RfbBasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=False, sn=False, bias=False):
        super(RfbBasicConv, self).__init__()

        assert (bn and sn) is not True
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.sn = nn.InstanceNorm2d(out_planes, affine=True) if sn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.sn is not None:
            x = self.sn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class RfbBasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False,
                 sn=False, bias=False):
        super(RfbBasicSepConv, self).__init__()
        assert (bn and sn) is not True

        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.sn = nn.InstanceNorm2d(in_planes, affine=True) if sn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.sn is not None:
            x = self.sn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB_S(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB_S, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 4

        self.branch0 = nn.Sequential(
            RfbBasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            RfbBasicSepConv(inter_planes, kernel_size=3, stride=1, padding=1, dilation=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            RfbBasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            RfbBasicConv(inter_planes, inter_planes, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            RfbBasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            RfbBasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            RfbBasicConv(inter_planes, inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            RfbBasicSepConv(inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch3 = nn.Sequential(
            RfbBasicConv(in_planes, inter_planes // 2, kernel_size=1, stride=1),
            RfbBasicConv(inter_planes // 2, (inter_planes // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            RfbBasicConv((inter_planes // 4) * 3, inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            RfbBasicSepConv(inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = RfbBasicConv(4 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.ConvLinear(out)
        out = out * self.scale + x
        out = self.relu(out)

        return out


# +++++++++++++++++++++++++++++++++++++++ Histogram Layer +++++++++++++++++++++++++++++++++++++++++++++++#
class GaussianHistogram(nn.Module):
    """
    Use gaussian distribution
    Args:
        bins: number of bins to seperate values
        min: minium vale of the data
        max: maximum value of the data
        sigma: a learable paramerter, init=0.01
    """

    def __init__(self, bins, min, max, sigma, require_grad=False):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max

        self.sigma = torch.tensor([sigma])
        self.sigma = Variable(self.sigma, requires_grad=require_grad)

        self.delta = float(max - min) / float(bins)
        self.centers = nn.Parameter(float(min) + self.delta * (torch.arange(bins).float() + 0.5), requires_grad=False)

    def forward(self, x, attention_mask=None):
        device = x.device
        self.sigma = self.sigma.to(device)
        self.centers = self.centers.to(device)

        x = torch.unsqueeze(x, dim=1) - torch.unsqueeze(self.centers, 1)
        hist_dist = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        # multiply with attention mask
        if not type(attention_mask) == type(None):
            hist_dist *= torch.unsqueeze(attention_mask, 1)

        hist = hist_dist.sum(dim=-1)
        hist = hist / torch.sum(hist, dim=1, keepdim=True)

        return hist, hist_dist


# +++++++++++++++++++++++++++++++++++++++ Attentioned Layer +++++++++++++++++++++++++++++++++++++++++++++++#
class AttentionExtractModule(ResNet):
    """
    Attention map extraction of ResNet-34
    """
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        g0 = self.layer1(x)
        g1 = self.layer2(g0)
        g2 = self.layer3(g1)
        g3 = self.layer4(g2)

        return [g.pow(2).mean(1) for g in (g0, g1, g2, g3)], [g0, g1, g2, g3]

if __name__ == '__main__':
    data = 1.0 * torch.ones((2, 256, 400))
    mean = torch.ones(2, 1, 400) * 2.3
    t = data * mean
    print(data - mean)