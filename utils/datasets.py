import glob
import random
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = 


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
                            for path in self.img_files]

        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.length = len(self.img_files)

    def __getitem__(self, index):
        # 读取图片
        img_path = self.img_files[index % self.length].rstrip()
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # 处理图片，维度至少为3
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

