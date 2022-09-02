#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/9/2 21:56
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : train.py
@Project    : FaceScore
@Description:
"""
import torch.nn as nn
import math
import pickle
import torch
import numpy as np
import torch
import logging
import os
import datetime
from FaceData import FaceDataset
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models

logdir = './logs'
log_file_name = 'PointNet-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
log_path = os.path.join(logdir, log_file_name + '.log')

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)  # 将日志的输出级别调节为info
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
# 清除原來自帶的root
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(fmt=formatter)

file_handler = logging.FileHandler(filename=log_path, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt=formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
# ******日志相关部分结束******

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
# 修改全连接层的输出
num_ftrs = model.fc.in_features
# 十分类，将输出层修改成10
model.fc = nn.Linear(num_ftrs, 1)
batch_size = 16
num_workers=8
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.ToTensor(),  # 转化为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 进行归一化
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 进行归一化
])
train_set = FaceDataset(img_path='/home/chase/Boyka/SCUT-FBP5500_v2/Images',
                        label_path='/home/chase/Boyka/SCUT-FBP5500_v2/train_test_files/split64/train.txt',
                        transform=train_transforms)
test_set = FaceDataset(img_path='/home/chase/Boyka/SCUT-FBP5500_v2/Images',
                       label_path='/home/chase/Boyka/SCUT-FBP5500_v2/train_test_files/split64/test.txt',
                       transform=test_transforms)
train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
test_loader = data.DataLoader(dataset=test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)

num_epochs = 30
num = 20
criterion = nn.MSELoss()
net = model
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
for epoch in range(num_epochs):
    run_loss, total_loss = 0.0, 0.0
    run_ae, total_ae = 0.0, 0.0
    test_ae = 0.0
    gt = np.array([])
    pre = np.array([])
    # 训练
    net.train()
    for i, data in enumerate(train_loader):
        obj, label = data
        obj, label = obj.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = net(obj)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        ae = torch.mean(torch.abs(outputs - label))
        run_ae += ae.item()

        if i % num == 0:
            logger.info(
                'train epoch %d, %2d loss: %.3f, train mae: %.3f' % (epoch, i / num, run_loss / num, run_ae / num))
            run_loss = 0
            run_ae = 0
        total_loss += loss.item()
        total_ae += ae.item()
    logger.info("epoch %d train mae: %.3f " % (epoch, total_ae / len(train_loader)))
    # 测试
    net.eval()
    for i, data in enumerate(test_loader):
        obj, label = data
        obj, label = obj.to(device), label.to(device)
        outputs = net(obj)
        gt = np.hstack((gt, label.cpu().detach().numpy().squeeze()))
        pre = np.hstack((pre, outputs.cpu().detach().numpy().squeeze()))
        ae = torch.mean(torch.abs(outputs - label))
        test_ae += ae.item()

    logger.info("epoch %d test mae: %.3f ：" % (epoch, test_ae / len(test_loader)))
    # logger.info("epoch %d test corrcoef: %.3f ：" % (epoch, pc(gt, pre)))
    logger.info("epoch %d test corrcoef: %.3f ：" % (epoch, np.corrcoef(gt, pre)[0][1]))
torch.save(net.state_dict(), os.path.join('./models', log_file_name + '.pth'))

print("finished training!")
