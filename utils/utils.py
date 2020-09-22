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
    # wh2代表的是一个ground truth
    # wh1代表的是所有的候选框，所以维度比wh1高一维；wh2.t()可以升一维度
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]

    # inter_area是将两个方框固定在同一个左上角，计算重叠距离
    # union_area是计算两个方框的面积并相加，并减去重叠的部分
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    pass


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
    # targets维度的含义可从datasets.py中获得，共6维；0代表当前batch第几张照片，1类别，2:6为bbox
    target_boxes = target[:, 2:6] * nG

    # 获取ground truth对应的x,y和w,h
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]

    # 匹配候选框和ground truth，计算IOU，并得到最佳IOU
    # ious [batch_size, anchors, gt]，每一列代表不同的gt，在所有anchors上的交并比大小
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])

    # .max(0)代表获取每列的最大值；best_ious得到指定dim最大值，best_n代表索引
    # 即best_ious记录了每一列最大的交并比，best_n记录了其idx
    best_ious, best_n = ious.max(0)

    # 分离目标值，b代表idx，target_labels代表类别
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()

    # 设置掩码，best_n的维度是ground truth，所以与ground truth对应的那几个框的维度设置为1
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # 进一步修改noobj_mask，为后续损失做准备
    # 当iou超过ignore threshold时，将noobj_mask设置为0
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # 调整坐标
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

    # label的独热编码
    tcls[b, best_n, gj, gi, target_labels] = 1

    # 计算标签的正确性，以及anchor处的最佳iou
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

