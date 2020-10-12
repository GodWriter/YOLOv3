import random
import os
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset
from utils.augmentations import horisontal_flip


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest")
    return image


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # (左, 右, 上, 下)，当矮了(h < w)就上下补pad，高了(h > w)就左右补pad
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


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

        # ---------
        #  Image
        # ---------

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

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % self.length].rstrip()

        targets = None
        if os.path.exists(label_path):

            # boxes得初始维度为(1, 5)，分别为类别class和bbox坐标
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

            # 得到标签文件中未padding图像中，物体的相对坐标，并转化为原始坐标
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)

            # 调整得到padding之后的原始坐标
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]

            # 返回(x, y, w, h)，变换成相对尺寸的坐标了
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] = w_factor / padded_w
            boxes[:, 4] = h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):

        # 获取__getitem__()中，得到的三个变量
        paths, imgs, targets = list(zip(*batch))

        # 移除空得bbox，并给bbox编号，编号代表其属于第几张图片
        targets = [boxes for boxes in targets if boxes is not None]
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        # 将筛选后的bbox根据第0维拼接，即拼接成一个batch
        targets = torch.cat(targets, 0)

        # 每10个batch改变一下图像尺寸
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))

        # 将图像resize到相应得尺寸
        # 为什么改变了图像尺寸，却不需要改变bbox？因为bbox记录的是相对位置，所以保持不变
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
