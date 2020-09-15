from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.autograd import Variable
from utils.parse_config import *


def create_modules(module_defs):
    # 得到模型的配置参数
    hyperparams = module_defs.pop(0)

    # 得到输出的通道数
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()

    for module_i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2

            modules.add_module(f"conv_{module_i}",
                               nn.Conv2d(in_channels=output_filters[-1],
                                         out_channels=filters,
                                         kernel_size=kernel_size,
                                         stride=int(module_def["stride"]),
                                         padding=pad,
                                         bias=not bn))

            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_def["type"] == "route":
            # 用于特征融合
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
            modules.add_module(f"route_{module_i}", EmptyLayer())

        elif module_def["type"] == "shortcut":
            # 用于残差网络
            filters = output_filters[1:][int(module_def["from"])]
            modules.add_module(f"shortcut_{module_i}", EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(',')]

            # 抽取anchors
            anchors = [int(x) for x in module_def["anchors"].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]

            # 其他信息
            num_classes = int(module_def["classes"])
            img_size = int(hyperparams["height"])

            # 定义检测层
            yolo_layer = YOLOLayer(anchors, num_classes, img_size)
            modules.add_module(f"yolo_{module_i}", yolo_layer)

        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()

        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class EmptyLayer(nn.Module):
    """ route和shortcut的占位符"""

    def __init__(self):
        super(EmptyLayer, self).__init__()


class YOLOLayer(nn.Module):
    """检测层"""

    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()

        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def compute_grid_offsets(self, grid_size, cuda=True):
        pass

    def forward(self, x, targets=None, img_dim=None):
        # 定义Tensors的种类
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        # x [batch, channel, width, height]
        self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        # channel = num_anchors * (num_classes + 5)
        # prediction [batch, num_anchors, num_classes+5, width, height]
        # prediction [batch, num_anchors, width, height, num_classes+5]
        prediction = (x.view(num_samples, self.num_anchors, self.num_classes+5, grid_size, grid_size)
                      .permute(0, 1, 3, 4, 2)
                      .contiguous())

        # x, y, w, h [batch, num_anchors, width, height]
        x = torch.sigmoid(prediction[..., 0]) # Center x
        y = torch.sigmoid(prediction[..., 1]) # Center y
        w = prediction[..., 2] # Width
        h = prediction[..., 3] # Height

        # pred_conf, pred_cls [batch, num_anchors, width, height, num_classes]
        pred_conf = torch.sigmoid(prediction[..., 4]) # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:]) # Cls pred




class Darknet(nn.Module):
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()

        self.module_def = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_def)

        pass
