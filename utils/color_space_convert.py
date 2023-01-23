"""
differentiable color space transfer function
"""
import numpy as np
import torch
import torch.nn as nn

from skimage.color import rgb2lab, lab2rgb

pi = torch.tensor(3.14159265358979323846)


class RgbToHsv(nn.Module):
    r"""Convert image from RGB to HSV.

    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to HSV.

    returns:
        torch.tensor: HSV version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    Examples::

        input = torch.rand(2, 3, 4, 5)
        hsv = kornia.color.RgbToHsv()
        output = hsv(input)  # 2x3x4x5

    """

    def __init__(self, training=True):
        super(RgbToHsv, self).__init__()
        self.training = training

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_hsv(image, self.training)


def rgb_to_hsv(image: torch.Tensor, training) -> torch.Tensor:
    r"""Convert an RGB image to HSV.

    Args:
        input (torch.Tensor): RGB Image to be converted to HSV.

    Returns:
        torch.Tensor: HSV version of the image.
        :param image:
        :param training:
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    maxc: torch.Tensor = image.max(-3)[0]
    minc: torch.Tensor = image.min(-3)[0]

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = deltac  # / v let's use chromic to replace saturatoin

    s[torch.isnan(s)] = 0.

    # avoid division by zero
    deltac = torch.where(
        deltac == 0, torch.ones_like(deltac), deltac)

    rc: torch.Tensor = (maxc - r) / deltac
    gc: torch.Tensor = (maxc - g) / deltac
    bc: torch.Tensor = (maxc - b) / deltac

    maxg: torch.Tensor = g == maxc
    maxr: torch.Tensor = r == maxc

    h: torch.Tensor = 4.0 + gc - rc
    h[maxg] = 2.0 + rc[maxg] - bc[maxg]
    h[maxr] = bc[maxr] - gc[maxr]
    h[minc == maxc] = 0.0

    h = (h / 6.0) % 1.0
    # h = 2 * pi.to(image.device) * h
    if training:
        return torch.stack([h, s, v], dim=-3)
    else:
        return torch.unsqueeze(v, 1)


class RgbToYuv(nn.Module):
    r"""Convert image from RGB to YUV
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB image to be converted to YUV.
    returns:
        torch.tensor: YUV version of the image.
    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`
    Reference::
        [1] https://es.wikipedia.org/wiki/YUV#RGB_a_Y'UV
    """

    def __init__(self) -> None:
        super(RgbToYuv, self).__init__()

    def forward(  # type: ignore
            self, input: torch.Tensor) -> torch.Tensor:
        return rgb_to_yuv(input)


def rgb_to_yuv(input: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): RGB Image to be converted to YUV.
    Returns:
        torch.Tensor: YUV version of the image.
    See :class:`~kornia.color.RgbToYuv` for details."""
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {type(input)}")

    if not(len(input.shape) == 3 or len(input.shape) == 4):
        raise ValueError(f"Input size must have a shape of (*, 3, H, W) or (3, H, W). Got {input.shape}")

    if input.shape[-3] != 3:
        raise ValueError(f"Expected input to have 3 channels, got {input.shape[-3]}")

    r, g, b = torch.chunk(input, chunks=3, dim=-3)
    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b
    yuv_img: torch.Tensor = torch.cat((y, u, v), -3)
    return yuv_img


def l_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    L = (L + 1.) * 50.
    L = L/100.
    rgb_imgs = L.repeat(1,3,1,1).cpu().detach().numpy()
    return torch.from_numpy(rgb_imgs)


def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().detach().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)

    rgb_imgs = np.stack(rgb_imgs, axis=0)
    rgb_imgs = rgb_imgs.transpose((0, 3, 1, 2))
    rgb_imgs = np.asarray(rgb_imgs, dtype=np.float32)

    return torch.from_numpy(rgb_imgs)