import os
import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from nets.frcnn import FasterRCNN
from nets.frcnn_training import (FasterRCNNTrainer, get_lr_scheduler,
                                 set_optimizer_lr, weights_init)
from utils.utils import get_classes, show_config
from nets.classifier import Resnet50RoIHead, VGG16RoIHead
from nets.resnet50 import resnet50
from tools.mist_train import mist_train
from tools.surtooldataset import SurtoolDataset


if __name__ == "__main__":
    Cuda = True if torch.cuda.is_available() else False
    train_gpu = [0, ]
    classes_path = 'surtool_model/classes.txt'
    model_path = 'logs/weights_7classes.pth'
    input_shape = [512,512]
    backbone = "resnet50"
    pretrained = True
    anchors_size = [8, 16, 32]
    batch_size = 1
    epoches = 100

    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)

    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)
    ngpus_per_node = len(train_gpu)
    print('Number of devices: {}'.format(ngpus_per_node))

    # 先load原来预训练的7个类的模型
    model = FasterRCNN(7, anchor_scales=anchors_size, backbone=backbone, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

        # 重新初始化head为14类的分类器
        _, classifier = resnet50(pretrained)

        model.head = Resnet50RoIHead(
                n_class         = num_classes + 1,
                roi_size        = 14,
                spatial_scale   = 1,
                classifier      = classifier
            )


    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model_train)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    #初始化dataloader
    # 创建数据集
    train_data = SurtoolDataset('../data/SurtoolDataset/train.txt', '../data/SurtoolDataset/labels.csv',
                                transform=None)
    test_data = SurtoolDataset('../data/SurtoolDataset/test.txt', '../data/SurtoolDataset/labels.csv',
                               transform=None)

    # 创建数据加载器
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 生成假标签自训练
    mist_train(model, train_loader, epoches=epoches)
