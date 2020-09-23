from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.autograd import Variable
from utils.parse_config import *
from utils.utils import build_targets, to_cpu


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
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # gird_size即为 FPN输出的大小，有 13*13, 26*26, 52*52
        # stride即为 img_dim缩放到当前特征图大小后，需缩小的倍数
        self.grid_size = grid_size
        self.stride = self.img_dim / self.grid_size
        g = self.grid_size

        # 计算特征网格的坐标
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)

        # 计算anchor经过同等比例缩放后的大小，并分别保存宽度和高度
        # scaled_anchors的有效维度只有2维
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

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

        # 如果grid_size改变了，即处理不同尺度的特征图了，需要根据新的特征图计算偏置等信息
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # 对应论文中预测框的计算
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        # [num_samples, -1, 4]，可得到所有预测的bbox，统一显示，其他同理
        output = torch.cat((pred_boxes.view(num_samples, -1, 4) * self.stride,
                            pred_conf.view(num_samples, -1, 1),
                            pred_cls.view(num_samples, -1, self.num_classes)), -1)

        # target是什么？暂时没弄清楚
        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(pred_boxes=pred_boxes,
                                                                                                      pred_cls=pred_cls,
                                                                                                      target=targets,
                                                                                                      anchors=self.scaled_anchors,
                                                                                                      ignore_thres=self.ignore_thres)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])

            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()

            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()

            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {"loss": to_cpu(total_loss).item(),
                            "x": to_cpu(loss_x).item(),
                            "y": to_cpu(loss_y).item(),
                            "w": to_cpu(loss_w).item(),
                            "h": to_cpu(loss_h).item(),
                            "conf": to_cpu(loss_conf).item(),
                            "cls": to_cpu(loss_cls).item(),
                            "cls_acc": to_cpu(cls_acc).item(),
                            "recall50": to_cpu(recall50).item(),
                            "recall75": to_cpu(recall75).item(),
                            "precision": to_cpu(precision).item(),
                            "conf_obj": to_cpu(conf_obj).item(),
                            "conf_noobj": to_cpu(conf_noobj).item(),
                            "grid_size": grid_size}

            return output, total_loss


class Darknet(nn.Module):
    def __init__(self, config_path, img_size=416):
        super(Darknet, self).__init__()

        self.module_def = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_modules(self.module_def)

        pass
