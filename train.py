import os
import torch
import argparse

from models import *
from utils.utils import *
from utils.logger import *
from utils.datasets import *
from utils.parse_config import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集配置
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["name"])

    # 模型初始化
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # 特殊情况下从checkpoint开始训练
    if opt.pretrained_weights:
        pass

    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.n_cpu,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    optimizer = torch.optim.Adam(model.parameters())

    metrics = ["grid_size",
               "loss",
               "x",
               "y",
               "w",
               "h",
               "conf",
               "cls",
               "cls_acc",
               "recall50",
               "recall75",
               "precision",
               "conf_obj",
               "conf_noobj"]

    for epoch in range(opt.epoches):
        pass
