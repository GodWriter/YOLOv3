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

        if module_defs["type"] == "convolutional":
            bn = int(module_defs["batch_normalize"])
            filters = int(module_defs["filters"])
            kernel_size = int(module_defs["size"])
            pad = (kernel_size - 1) // 2

            modules.add_module(f"conv_{module_i}",
                               nn.Conv2d(in_channels=output_filters[-1],
                                         out_channels=filters,
                                         kernel_size=kernel_size,
                                         stride=int(module_defs["stride"]),
                                         padding=pad,
                                         bias=not bn))

            if bn:
                modules.add_module(f"batch_norm_{module_i}", nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5))
            if module_defs["activation"] == "leaky":
                modules.add_module(f"leaky_{module_i}", nn.LeakyReLU(0.1))

        elif module_defs["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_defs["stride"]), mode="nearest")
            modules.add_module(f"upsample_{module_i}", upsample)

        elif module_defs["type"] == "route":
            layers = [int(x) for x in module_defs["layers"].split(",")]
            filters = sum([output_filters[1:][i] for i in layers])
        elif module_defs["type"] == "shortcut":
            pass
        elif module_defs["type"] == "yolo":
            pass

    return hyperparams, module_list


class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Darknet(nn.Module):
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)

        pass
