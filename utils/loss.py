"""
Loss functions
"""
from fastai.torch_core import *
from fastai.callbacks import hook_outputs

import numpy as np
import torch.nn as nn
from utils.ssim import SSIM
import torchvision.models as models
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from models import networks
from utils.lpips_pytorch import LPIPS, lpips
from utils.color_space_convert import RgbToHsv, RgbToYuv


class PerceptualLoss(nn.Module):
    """
    VGG16-Based Perceptual Loss
    Args:
        layer_wgts: weights of loss for different pretrained layer
    """

    def __init__(self, layer_wgts=[10, 20, 70, 10]):
        super().__init__()

        self.m_feat = models.vgg16_bn(pretrained=True).features.cuda().eval()
        requires_grad(self.m_feat, False)

        blocks = [
            i - 1
            for i, o in enumerate(children(self.m_feat))
            if isinstance(o, nn.MaxPool2d)
        ]

        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts[1:]
        self.metric_names = ['pixel'] + [f'feat_{i}' for i in range(len(layer_ids))]
        self.base_loss = F.l1_loss
        self.base_weight = layer_wgts[0]
        self.feat_losses = None
        self.metrics = None

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        if input.shape[1] == 1:
            input = input.repeat(1, 3, 1, 1)
        if target.shape[1] == 1:
            target = target.repeat(1, 3, 1, 1)

        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_weight * self.base_loss(input, target)]
        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()


class WassersteinLoss(nn.Module):
    """
    Wassersterin loss
    """

    def __init__(self):
        super().__init__()

    def forward(self, score, real, gen):
        """
        During training discriminator, we want the score low for real, high for fake,
        so loss should multply with 1 for real, -1 for fake. During training generator,
        we want the score as low as possible, so multiply with 1
        :param score: discriminator continuous score.
        :param real: whether this is a real example or fake one
        :param gen:  this is during training generator or discriminator
        :return:
        """
        device = score.device
        one = torch.ones_like(score, dtype=score.dtype)
        mone = one * -1
        if gen:
            return score * one.to(device)
        else:
            if real:
                return score * one.to(device)
            else:
                return score * mone.to(device)


class EarthMoverDisteLoss(nn.Module):
    """
    Earth Mover Distance Loss
    Args:

    """

    def __init__(self):
        super().__init__()
        self.creterion = nn.MSELoss()

    def forward(self, input, target):
        """
        Loss calculation
        :param input:  input histogram, shape required: (N, K)
        :param target: target histogram, shape required: (N, K)
        :return:
        """
        input_cumsum = torch.cumsum(input, dim=1)
        target_cumsum = torch.cumsum(target, dim=1)
        return self.creterion(input_cumsum, target_cumsum)


class HistogramLoss(nn.Module):
    """
    Calculate histogram distribution loss #TODO: Make RGB also avaialble, right now only yuv supported
    Args:
    """

    def __init__(self):
        super().__init__()
        self.creterion = EarthMoverDisteLoss()
        self.histlayer = networks.GaussianHistogram(bins=256, min=0, max=1, sigma=0.01)

    def forward(self, input, target):
        channels = input.shape[1]
        losses = []
        for i in range(channels):
            input_channel = torch.flatten(input[:, i, :, :], start_dim=1, end_dim=-1)
            target_channel = torch.flatten(target[:, i, :, :], start_dim=1, end_dim=-1)

            input_hist, _ = self.histlayer(input_channel)
            target_hist, _ = self.histlayer(target_channel)
            losses.append(self.creterion(input_hist, target_hist))

        return sum(losses)


def histogram(args):
    """
    Hisgrogram distribution loss
    :param args:
    :return:
    """
    return HistogramLoss()


def lipis_eval(net_type='alex'):
    """
    LIPIS metric
    :param net_type: supported: alexnet, vgg16 and squeezenet
    :return:
    """
    return LPIPS(net_type=net_type, version='0.1')


def wasserstein(args):
    """
    wassertein loss
    :param args:
    :return:
    """
    return WassersteinLoss()


def perceptual(args):
    """
    Perceptual loss
    :param args:
    :return:
    """
    return PerceptualLoss(layer_wgts=args['layer_weights'])


def ssim(args=None):
    """
    SSIM loss
    :return:
    """
    return SSIM()


def mse(args=None):
    """
    MSE loss
    :return:
    """
    return nn.MSELoss()


def l1(args=None):
    """
    L1 loss
    :param args:
    :return:
    """
    return nn.L1Loss()


def intermediate_loss(hypes, layer, loss_func, predict_batch, target_batch):
    """
    Calculate the intermediate loss based on loss name
    :param hypes:  hype yaml dict
    :param layer:  intermediate layer name
    :param loss_func:  loss function
    :param predict_batch:  intermediate output
    :param target_batch:  target output
    :return:
    """
    if not hypes['color_pretrain']['flag']:
        downsampled_target_batch = F.interpolate(target_batch,
                                                 scale_factor=1 / hypes['arch']['args']['scale'],
                                                 mode='bicubic')
    else:
        downsampled_target_batch = target_batch

    if 'rgb' in layer:
        return loss_func(predict_batch, downsampled_target_batch)

    if 'gray' in layer:
        downsampled_target_batch = torch.mean(downsampled_target_batch, dim=1)
        downsampled_target_batch = downsampled_target_batch.unsqueeze(dim=1)

        predict_batch = torch.mean(predict_batch, dim=1)
        predict_batch = predict_batch.unsqueeze(dim=1)
        return loss_func(predict_batch, downsampled_target_batch)

    if 'hsv' in layer:
        hsv = RgbToHsv()
        downsampled_target_batch_hsv = hsv(downsampled_target_batch)
        predict_batch_hsv = hsv(predict_batch)
        return loss_func(predict_batch_hsv[:, :2, :, :], downsampled_target_batch_hsv[:, :2, :, :])

    if 'yuv' in layer:
        yuv = RgbToYuv()
        downsampled_target_batch_yuv = yuv(downsampled_target_batch)
        predict_batch_yuv = yuv(predict_batch)
        return loss_func(predict_batch_yuv[:, 1:, :, :], downsampled_target_batch_yuv[:, 1:, :, :])


def loss_sum(hypes, creterion, predict_dict, target_batch, ref_batch=None):
    """
    Sum up all loss with their weights
    :param creterion: dictionary of all loss functions
    :param hypes: yaml dict
    :param predict_dict:  prediction dictionary
    :param target_batch:  groundtruth
    :param ref_batch: reference batch
    :return:
    """
    # final output
    predict_batch = predict_dict['output']
    final_loss = None
    loss_param = hypes['train_params']['loss']
    for name, loss_func in creterion.items():
        if name not in loss_param:
            continue
        current_loss = loss_func(predict_batch, target_batch)
        # ssim need to be reversed
        if name == 'ssim':
            current_loss = 1 - current_loss
        if not final_loss:
            final_loss = loss_param[name]['weight'] * current_loss
        else:
            final_loss += loss_param[name]['weight'] * current_loss

    if 'intermediate' in hypes['train_params']:
        intermediate_dict = hypes['train_params']['intermediate']
        for layer in intermediate_dict['layer']:
            predict_batch = predict_dict['aux']
            for name, loss_func in creterion.items():
                if 'intermediate' not in name:
                    continue
                func_origin_name = name.replace('intermediate_', '')
                current_loss = intermediate_dict['loss'][func_origin_name]['weight'] * \
                               intermediate_loss(hypes,
                                                 layer,
                                                 loss_func,
                                                 predict_batch,
                                                 target_batch)
                if func_origin_name == 'ssim':
                    current_loss = intermediate_dict['loss'][func_origin_name]['weight'] - current_loss

                final_loss += current_loss
    return final_loss


def loss_sum_gan(hypes, creterion, score, real, gen):
    """
    get gan loss
    :param hypes:
    :param creterion:
    :param score:
    :param real:
    :param gen:
    :return:
    """
    final_loss = None
    loss_param = hypes['gan']['loss']
    for name, loss_func in creterion.items():
        if name not in loss_param:
            continue
        current_loss = loss_func(score, real, gen) * \
                       (loss_param[name]['gen_weight'] if gen else loss_param[name]['dis_weight'])
        if not final_loss:
            final_loss = current_loss
        else:
            final_loss += current_loss

    return final_loss


def batch_psnr(img, imclean, data_range):
    """
    Compute psnr of batches
    :param img:  prediction
    :param imclean:  target image
    :param data_range:  maximum value
    :return:
    """
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return PSNR / Img.shape[0]
