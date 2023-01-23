"""
Customized layer for non-parametric operation
"""
import torch
import cv2
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


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


class GaussianHistogram(nn.Module):
    """
    Use gaussian distribution
    Args:
        bins: number of bins to seperate values
        min: minium vale of the data
        max: maximum value of the data
        sigma: a learable paramerter, init=0.01
    """

    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max

        self.sigma = torch.tensor([sigma])
        self.sigma = Variable(self.sigma, requires_grad=False)

        self.delta = float(max - min) / float(bins)
        self.centers = nn.Parameter(float(min) + self.delta * (torch.arange(bins).float() + 0.5), requires_grad=False)

    def forward(self, x):
        device = x.device
        self.sigma = self.sigma.to(device)
        self.centers = self.centers.to(device)

        x = torch.unsqueeze(x, dim=1) - torch.unsqueeze(self.centers, 1)
        hist_dist = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        hist = hist_dist.sum(dim=-1)
        hist = hist / torch.sum(hist, dim=1, keepdim=True)

        return hist, hist_dist


if __name__ == '__main__':
    dims = 3
    data = 1 + torch.randn((dims, 224, 224))
    data = torch.flatten(data, start_dim=1, end_dim=-1)

    data2 = 0.5 + torch.randn((dims, 224, 224))
    data2 = torch.flatten(data2, start_dim=1, end_dim=-1)

    data3 = 6 + torch.randn((dims, 224, 224))
    data3 = torch.flatten(data3, start_dim=1, end_dim=-1)

    emd_loss = EarthMoverDisteLoss()
    gausshist = GaussianHistogram(bins=256, min=0, max=3, sigma=0.01)

    hist1, hist1_dist = gausshist(data)
    hist2, hist2_dist = gausshist(data2)
    hist3, hist3_dist = gausshist(data3)
    # joint distribution
    hist_join_12 = torch.matmul(hist1_dist, hist2_dist.permute(0, 2, 1)) / (224 * 224)
    loss2 = emd_loss(hist1, hist2)
    print(loss2)

    hist_join_13 = torch.matmul(hist1_dist, hist3_dist.permute(0, 2, 1)) / (224 * 224)
    loss3 = emd_loss(hist1, hist3)
    print(loss3)

    print(torch.sum(hist_join_12, dim=(1, 2)))
    print(torch.sum(hist_join_13, dim=(1, 2)))
