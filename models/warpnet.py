"""
Calculate the similarity map between the reference and input under different levels and calculate the
simalirity map
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torch_models
from torchvision.models import resnet34
from torchvision.models.resnet import ResNet, BasicBlock

from models.networks import GaussianHistogram, AttentionExtractModule
from models.dense121_unet_histogram_attention import HistogramLayerLocal


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ResidualBlock, self).__init__()
        self.padding1 = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.padding2 = nn.ReflectionPad2d(padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride)
        self.bn2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.padding1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.padding2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.prelu(out)
        return out


def padding_customize(x1, x2):

    diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
    diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
    # in case input size are odd
    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2], mode='replicate')

    return x1


class WarpNet(nn.Module):
    """
    Inputs are the res34 features
    """
    def __init__(self, feat1=64, feat2=128, feat3=256, feat4=512):
        super(WarpNet, self).__init__()
        self.feature_channel = 64
        self.in_channels = self.feature_channel * 4
        self.inter_channels = 256
        # 44*44
        self.layer2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(feat1, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=2),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )

        self.layer3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(feat2, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )

        # 22*22->44*44
        self.layer4_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(feat3, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )

        # 11*11->44*44
        self.layer5_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(feat4, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )

        self.layer = nn.Sequential(
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
        )

        self.theta = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )
        self.phi = nn.Conv2d(
            in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        self.upsampling = nn.Upsample(scale_factor=4)

    def forward(
        self,
        B_hist,
        A_relu2_1,
        A_relu3_1,
        A_relu4_1,
        A_relu5_1,
        B_relu2_1,
        B_relu3_1,
        B_relu4_1,
        B_relu5_1,
        temperature=0.001 * 5,
        detach_flag=False,
    ):
        batch_size = B_hist.shape[0]

        # scale feature size to 44*44
        A_feature2_1 = self.layer2_1(A_relu2_1)
        B_feature2_1 = self.layer2_1(B_relu2_1)
        A_feature3_1 = self.layer3_1(A_relu3_1)
        B_feature3_1 = self.layer3_1(B_relu3_1)
        A_feature4_1 = self.layer4_1(A_relu4_1)
        B_feature4_1 = self.layer4_1(B_relu4_1)
        A_feature5_1 = self.layer5_1(A_relu5_1)
        B_feature5_1 = self.layer5_1(B_relu5_1)

        # concatenate features
        if A_feature5_1.shape[2] != A_feature2_1.shape[2] or A_feature5_1.shape[3] != A_feature2_1.shape[3]:
            A_feature2_1 = padding_customize(A_feature2_1, A_feature5_1)
            A_feature3_1 = padding_customize(A_feature3_1, A_feature5_1)
            A_feature4_1 = padding_customize(A_feature4_1, A_feature5_1)

        if B_feature5_1.shape[2] != B_feature2_1.shape[2] or B_feature5_1.shape[3] != B_feature2_1.shape[3]:
            B_feature2_1 = padding_customize(B_feature2_1, B_feature5_1)
            B_feature3_1 = padding_customize(B_feature3_1, B_feature5_1)
            B_feature4_1 = padding_customize(B_feature4_1, B_feature5_1)

        A_features = self.layer(torch.cat((A_feature2_1, A_feature3_1, A_feature4_1, A_feature5_1), 1))
        B_features = self.layer(torch.cat((B_feature2_1, B_feature3_1, B_feature4_1, B_feature5_1), 1))

        # pairwise cosine similarity
        theta = self.theta(A_features).view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        theta = theta - theta.mean(dim=-1, keepdim=True)  # center the feature
        theta_norm = torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon
        theta = torch.div(theta, theta_norm)
        theta_permute = theta.permute(0, 2, 1)  # 2*(feature_height*feature_width)*256
        phi = self.phi(B_features).view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        phi = phi - phi.mean(dim=-1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)
        f = torch.matmul(theta_permute, phi)  # 2*(feature_height*feature_width)*(feature_height*feature_width)
        if detach_flag:
            f = f.detach()

        f_similarity = f.unsqueeze_(dim=1)
        similarity_map = torch.max(f_similarity, -1, keepdim=True)[0]
        similarity_map = similarity_map.view(batch_size, 1, A_feature2_1.shape[2],  A_feature2_1.shape[3])

        # f can be negative
        f_WTA = f
        f_WTA = f_WTA / temperature
        f_div_C = F.softmax(f_WTA.squeeze_(), dim=-1)  # 2*1936*1936;

        # downsample the reference histogram
        feature_height, feature_width = B_hist.shape[2], B_hist.shape[3]
        B_hist = B_hist.view(batch_size, 512, -1)
        B_hist = B_hist.permute(0, 2, 1)
        y_hist = torch.matmul(f_div_C, B_hist)
        y_hist = y_hist.permute(0, 2, 1).contiguous()
        y_hist_1 = y_hist.view(batch_size, 512, feature_height, feature_width)

        # upsample, downspale the wrapped histogram feature for multi-level fusion
        upsample = nn.Upsample(scale_factor=2)
        y_hist_0 = upsample(y_hist_1)
        y_hist_2 = F.avg_pool2d(y_hist_1, 2)
        y_hist_3 = F.avg_pool2d(y_hist_1, 4)

        # do the same thing to similarity map
        similarity_map_0 = upsample(similarity_map)
        similarity_map_1 = similarity_map
        similarity_map_2 = F.avg_pool2d(similarity_map_1, 2)
        similarity_map_3 = F.avg_pool2d(similarity_map_1, 4)

        return [(y_hist_0, similarity_map_0), (y_hist_1, similarity_map_1),
                (y_hist_2, similarity_map_2), (y_hist_3, similarity_map_3)]


if __name__ == '__main__':
    data = torch.randn((2, 3, 256, 256))
    gt = torch.randn((2, 3, 256, 256))
    data = data.cuda()
    gt = gt.cuda()

    # res34 features
    base_resnet34 = resnet34(pretrained=True)
    att_model = AttentionExtractModule(BasicBlock, [3, 4, 6, 3])
    att_model.load_state_dict(base_resnet34.state_dict())
    att_model.cuda()
    att_model.eval()

    _, gt_features = att_model(gt)
    _, data_features = att_model(data)

    # Resize image by 4/8/16/32
    gt_resize_by_8 = F.adaptive_avg_pool2d(gt, (gt.shape[2] // 8, gt.shape[3] // 8))

    # histogram layer
    hist_layer = HistogramLayerLocal()
    hist_layer.cuda()

    gt_hist = hist_layer(gt_resize_by_8, gt_resize_by_8)

    warpnet = WarpNet()
    warpnet.cuda()
    output = warpnet(gt_hist,
                     data_features[0], data_features[1], data_features[2], data_features[3],
                     gt_features[0], gt_features[1], gt_features[2], gt_features[3])
