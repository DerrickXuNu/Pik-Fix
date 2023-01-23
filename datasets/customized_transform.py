"""
Classes for customized transform for our data
"""
from random import randint

import cv2
import numpy as np
import torch
import imgaug.augmenters as iaa

from skimage.color import rgb2lab, lab2rgb
from utils.texture_libs import crack_generate, dust_generate
from utils.damage_libs import damage_generate


class CrackGenerator(object):
    """
    crack generation
    """

    def __call__(self, sample):
        input_image, gt_image = sample['input_image'], sample['gt_image']
        # crack generation
        code = randint(1, 12)
        processed_image, _ = crack_generate(cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR).copy(), code)

        # dust generation
        code = randint(1, 8)
        processed_image = dust_generate(processed_image.copy(), code)

        processed_image = np.expand_dims(cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY), -1)
        sample.update({'input_image': processed_image})
        return sample


class DamageGenerator(object):
    """
    damage generation
    """
    def __call__(self, sample):
        input_image = sample['input_image']
        processed_image = damage_generate(input_image.copy(), threash=20)
        sample.update({'input_image': processed_image})
        return sample



class RandomCrop(object):
    """
    Crop both input image and ground truth
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=256):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        input_image, gt_image = sample['input_image'], sample['gt_image']

        # make sure crop size is smaller than image size
        h, w = input_image.shape[:2]
        new_h, new_w = self.output_size

        if h < new_h:
            gt_image = cv2.resize(gt_image, None, fx=new_h / h, fy=new_h / h)
            input_image = cv2.resize(input_image, None, None, fx=new_h / h, fy=new_h / h)
        if w < new_w:
            gt_image = cv2.resize(gt_image, None, fx=new_w / w, fy=new_w / w)
            input_image = cv2.resize(input_image, None, fx=new_w / w, fy=new_w / w)

        assert gt_image.shape[0] >= new_h and gt_image.shape[1] >= new_w
        assert input_image.shape[0] >= new_h and input_image.shape[1] >= new_w

        # used for training as reference image
        ref_image = gt_image.copy() if 'ref_image' not in sample else sample['ref_image']
        ref_image = cv2.resize(ref_image, (gt_image.shape[1], gt_image.shape[0]))

        top = 0 if input_image.shape[0] == new_h else np.random.randint(0, input_image.shape[0] - new_h)
        left = 0 if input_image.shape[1] == new_w else np.random.randint(0, input_image.shape[1] - new_w)
        ref_image = ref_image[top: top + new_h, left: left + new_w]

        # generate random coordinates for cropping
        top = 0 if input_image.shape[0] == new_h else np.random.randint(0, input_image.shape[0] - new_h)
        left = 0 if input_image.shape[1] == new_w else np.random.randint(0, input_image.shape[1] - new_w)
        input_image = input_image[top: top + new_h,
                      left: left + new_w]
        gt_image = gt_image[top: top + new_h,
                   left: left + new_w]

        return {'input_image': input_image, 'gt_image': gt_image, 'ref_image': ref_image}


class RandomFlip(object):
    """
    Flip both input image and ground truth
    """

    def __call__(self, sample):
        input_image, gt_image = sample['input_image'], sample['gt_image']
        if len(input_image.shape) != 3:
            input_image = np.expand_dims(input_image, -1)

        if np.random.randint(2, size=1)[0] == 1:
            input_image = np.flip(input_image, axis=1)
            gt_image = np.flip(gt_image, axis=1)

        sample.update({'input_image': input_image.copy(), 'gt_image': gt_image.copy()})

        return sample


class RandomBlur(object):
    """
    Gaussian Blur and noise
    """

    def __call__(self, sample):
        input_image, gt_image = sample['input_image'], sample['gt_image']

        seq = iaa.Sequential([iaa.GaussianBlur(sigma=(0.0, 3.0)),
                              iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)])

        input_image = seq(image=input_image)
        sample.update({'input_image': input_image.copy(), 'gt_image': gt_image.copy()})

        return sample


class RandomAffine(object):
    """
    Apply Random Affine Transformation to the reference image
    """

    def __call__(self, sample):
        ref_image = sample['ref_image']
        aug = iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
        seq = iaa.Sequential([aug])
        ref_image = seq(image=ref_image)

        sample.update({'ref_image': ref_image})

        return sample


class RandomHueSaturation(object):
    """
    Add random hue and saturation to image
    """

    def __call__(self, sample):
        ref_image = sample['ref_image']
        aug = iaa.WithHueAndSaturation([
            iaa.WithChannels(0, iaa.Add((-20, 20))),
            iaa.WithChannels(1, [iaa.Multiply((0.8, 1.2)),
                                 iaa.LinearContrast((0.75, 1.25))])
        ])
        seq = iaa.Sequential([aug])
        ref_image = seq(image=ref_image)

        sample.update({'ref_image': ref_image})

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_image, gt_image = sample['input_image'], sample['gt_image']

        # the size has to be a integer mutiplier with 32
        height_mod = input_image.shape[0] % 32
        width_mod = input_image.shape[1] % 32

        if height_mod != 0 or width_mod != 0:
            height_residual = input_image.shape[0] // 32
            width_residual = input_image.shape[1] // 32

            input_image = input_image[:height_residual * 32, :width_residual * 32]
            gt_image = gt_image[:height_residual * 32, :width_residual * 32]

        if len(input_image.shape) != 3:
            input_image = np.expand_dims(input_image, -1)

        # We also need gray version for texture similarity
        ref_gray = cv2.cvtColor(gt_image, cv2.COLOR_RGB2GRAY)
        ref_gray = np.expand_dims(ref_gray, -1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.asarray(input_image, dtype=np.float32) / 255.

        gt_image = gt_image.transpose((2, 0, 1))
        gt_image = np.asarray(gt_image, dtype=np.float32) / 255.

        ref_gray = ref_gray.transpose((2, 0, 1))
        ref_gray = np.asarray(ref_gray, dtype=np.float32) / 255.

        if 'ref_image' in sample:
            ref_image = sample['ref_image']

            # We also need gray version for texture similarity
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
            ref_gray = np.expand_dims(ref_gray, -1)
            ref_gray = ref_gray.transpose((2, 0, 1))
            ref_gray = np.asarray(ref_gray, dtype=np.float32) / 255.

            ref_image = ref_image.transpose((2, 0, 1))
            ref_image = np.asarray(ref_image, dtype=np.float32) / 255.

            return {'input_image': torch.from_numpy(input_image),
                    'gt_image': torch.from_numpy(gt_image),
                    'ref_gray': torch.from_numpy(ref_gray),
                    'ref_image': torch.from_numpy(ref_image)}

        return {'input_image': torch.from_numpy(input_image),
                'gt_image': torch.from_numpy(gt_image),
                'ref_gray': torch.from_numpy(ref_gray), }


class TolABTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_image, gt_image = sample['input_image'], sample['gt_image']
        # the size has to be a integer mutiplier with 32
        height_mod = input_image.shape[0] % 32
        width_mod = input_image.shape[1] % 32

        if height_mod != 0 or width_mod != 0:
            height_residual = input_image.shape[0] // 32
            width_residual = input_image.shape[1] // 32

            input_image = input_image[:height_residual * 32, :width_residual * 32]
            gt_image = gt_image[:height_residual * 32, :width_residual * 32]

        if len(input_image.shape) != 3:
            input_image = np.expand_dims(input_image, -1)

        # Convert RGB To LAB
        input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
        input_lab = rgb2lab(input_image_rgb).astype("float32")
        input_L = np.expand_dims(input_lab[:, :, 0] / 50. - 1, -1)

        gt_image = rgb2lab(gt_image).astype("float32")
        gt_L = np.expand_dims(gt_image[:, :, 0] / 50. - 1, -1)
        gt_ab = gt_image[:, :, 1:] / 110.

        if 'ref_image' in sample:
            ref_image = sample['ref_image']
            ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY).copy()
            ref_gray = np.expand_dims(ref_gray, -1)
            ref_lab = rgb2lab(ref_image).astype("float32")
            ref_ab = ref_lab[:, :, 1:] / 110.

            ref_gray = ref_gray.transpose((2, 0, 1))
            ref_gray = np.asarray(ref_gray, dtype=np.float32) / 255.
            ref_ab = ref_ab.transpose((2, 0, 1))

        # transpose to torch tensor
        input_L = input_L.transpose((2, 0, 1))
        input_image = input_image.transpose((2, 0, 1))
        input_image = np.asarray(input_image, dtype=np.float32) / 255.

        gt_L = gt_L.transpose((2, 0, 1))
        gt_ab = gt_ab.transpose((2, 0, 1))

        if 'ref_image' in sample:
            return {'input_image': torch.from_numpy(input_image),
                    'input_L': torch.from_numpy(input_L),
                    'gt_L': torch.from_numpy(gt_L),
                    'gt_ab': torch.from_numpy(gt_ab),
                    'ref_ab': torch.from_numpy(ref_ab),
                    'ref_gray': torch.from_numpy(ref_gray)}
        else:
            return {'input_image': torch.from_numpy(input_image),
                    'input_L': torch.from_numpy(input_L),
                    'gt_L': torch.from_numpy(gt_L),
                    'gt_ab': torch.from_numpy(gt_ab)}


def rgbtolab(input_image):
    """
    Convert rgb2 lab.
    """
    # the size has to be a integer mutiplier with 32
    height_mod = input_image.shape[0] % 32
    width_mod = input_image.shape[1] % 32

    if height_mod != 0 or width_mod != 0:
        height_residual = input_image.shape[0] // 32
        width_residual = input_image.shape[1] // 32

        input_image = input_image[:height_residual * 32, :width_residual * 32]

    if len(input_image.shape) != 3:
        input_image = np.expand_dims(input_image, -1)

    # Convert RGB To LAB
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB)
    input_lab = rgb2lab(input_image_rgb).astype("float32")
    input_L = np.expand_dims(input_lab[:, :, 0] / 50. - 1, -1)

    # transpose to torch tensor
    input_L = input_L.transpose((2, 0, 1))
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.asarray(input_image, dtype=np.float32) / 255.

    return input_image, input_L
