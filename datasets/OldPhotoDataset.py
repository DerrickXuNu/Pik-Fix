"""
Customized dataset class for old photo
"""
from datasets.customized_transform import *

import os
import itertools
import json
import numpy as np
import random

import cv2
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.color_space_convert import lab_to_rgb


class OldPhotoDataset(Dataset):
    """
    Dataset should have a pair of data
    """

    def __init__(self, root_dir, transform=transforms.Compose([ToTensor()]), ref_json=False):
        """
        Args:
            :param root_dir: the path that contain all groundtruth and input images
            :param transform: callable function to do transform on origin data pair
            :param ref_json: whether load reference image from json
        """
        self.root_dir = root_dir
        self.gt_images = []
        self.ref_json_files = []
        self.ref_json = ref_json

        for folder in self.root_dir:
            gt_images = sorted([os.path.join(folder, x)
                                for x in os.listdir(folder) if x.endswith('.jpg') or x.endswith('.png')])
            if ref_json:
                ref_json_files = [os.path.join(folder, 'matches', os.path.split(x)[1][:-3] + 'json')
                                  for x in gt_images]
                self.ref_json_files += ref_json_files

            self.gt_images += gt_images

        self.transform = transform

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gt_image_name = self.gt_images[idx]

        gt_image = cv2.cvtColor(cv2.imread(gt_image_name), cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY), -1)

        data = {'input_image': input_image, 'gt_image': gt_image}

        if self.ref_json:
            gt_json_name = self.ref_json_files[idx]
            with open(gt_json_name, 'r') as f:
                match_json = json.load(f)

            random_seed = random.randint(0, len(match_json) - 1)
            ref_name = os.path.join(os.path.dirname(gt_image_name), match_json[random_seed]['name'] + '.jpg')
            ref_image = cv2.cvtColor(cv2.imread(ref_name), cv2.COLOR_BGR2RGB)

            data.update({'ref_image': ref_image})

        if self.transform:
            data = self.transform(data)

        data['image_name'] = gt_image_name
        return data


class RealOldPhotoDataset(Dataset):
    """
    Dataset should have a pair of data
    """

    def __init__(self, root_dir, transform=transforms.Compose([ToTensor()])):
        """
        Args:
            :param root_dir: the path that contain all groundtruth and input images
            :param transform: callable function to do transform on origin data pair
            :param ref_json: whether load reference image from json
        """
        self.root_dir = root_dir
        self.gt_images = []
        self.ref_json_files = []

        for folder in self.root_dir:
            gt_images = sorted([os.path.join(folder, x)
                                for x in os.listdir(folder) if 't' in x and 'm' not in x])

            self.gt_images += gt_images

        self.transform = transform

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gt_image_name = self.gt_images[idx]
        input_image_name = gt_image_name[:-5] + 'o.' + gt_image_name[-3:]

        gt_image = cv2.cvtColor(cv2.imread(gt_image_name), cv2.COLOR_BGR2RGB)
        input_image = cv2.imread(input_image_name, 0)
        input_image = np.expand_dims(input_image, -1)

        data = {'input_image': input_image, 'gt_image': gt_image}


        if self.transform:
            data = self.transform(data)

        data['image_name'] = gt_image_name
        return data


if __name__ == '__main__':
    oldphoto_dataset = RealOldPhotoDataset(root_dir=["../data/real_old_resize"],
                                       transform=transforms.Compose([TolABTensor()]))

    dataloader = DataLoader(oldphoto_dataset, batch_size=1, shuffle=False, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['input_image'].size(),
              sample_batched['gt_ab'].size())
        input_batch, input_l, gt_ab, gt_l = sample_batched['input_image'], sample_batched['input_L'], \
                                                              sample_batched['gt_ab'], sample_batched['gt_L'],
        gt_images = lab_to_rgb(gt_l, gt_ab)
        for i in range(1):
            input_images = input_batch[i].numpy().transpose(1, 2, 0)
            gt_image = gt_images[i].numpy().transpose(1, 2, 0)
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            # ref_gray = ref_grays[i].numpy().transpose(1, 2, 0)

            cv2.imshow('input_image', input_images)
            cv2.imshow('gt_image', gt_image)
            # cv2.imshow('ref_image', ref_gray)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
