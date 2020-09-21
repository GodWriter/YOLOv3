from __future__ import division

import torch


def load_classes(path):
    fp = open(path, 'r')
    names = fp.read().split('\n')[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]

    # inter_area是将两个方框固定在同一个左上角，计算重叠距离
    # union_area是计算两个方框的面积并相加
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0) # batch_size
    nA = pred_boxes.size(1) # 所有预测得到的bbox个数，即anchor
    nC = pred_cls.size(-1) # num_classes大小，即class
    nG = pred_boxes.size(2) # width和height，width = height，即grid_size

    # 输出的tensors，为什么要写成4维，是为了损失计算的方便性
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)

    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG).fill_(0)

    # 转换为与检测框相对应的位置
    # targets维度的含义可从datasets.py中获得，共6维；0占位，1类别，2:6为bbox
    target_boxes = target[:, 2:6] * nG

    # 获取ground truth对应的x,y和w,h
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # 匹配候选框和ground truth，计算IOU，并得到最佳IOU
    # .max(0)代表获取每列的最大值；best_ious得到指定dim最大值，best_n代表索引
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)

