#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/21 11:21
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : train.py
@Project    : FaceScore
@Description:
"""
# 神经网络
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import FaceDataset
import logging
import os
import torchvision.transforms as transforms
import torch.utils.data as data
from model import STN3d
from PointNet import PointNetReg
import datetime


def pc(x, y):
    n = len(x)
    sum_xy = np.sum(np.sum(x * y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x * x))
    sum_y2 = np.sum(np.sum(y * y))
    pc = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return pc


logdir = './logs'
log_file_name = 'PointNet-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
log_path = os.path.join(logdir, log_file_name + '.log')

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)  # 将日志的输出级别调节为info
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
# 清除原來自帶的root
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# 终端handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(fmt=formatter)

# 没有给handler指定日志级别，将使用logger的级别
file_handler = logging.FileHandler(filename=log_path, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt=formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)
# ******日志相关部分结束******

logger.addHandler(console_handler)
logger.addHandler(file_handler)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

criterion = nn.MSELoss()


transform = transforms.Compose(
    [transforms.ToTensor()
     ]
)
train_set = FaceDataset(obj_path='/home/chase/Boyka/SCUT-FBP5500_v2/images',
                        label_path='/home/chase/Boyka/SCUT-FBP5500_v2/train_test_files/split64/train.txt',
                        data_transform=transform, target_transform=transform)
test_set = FaceDataset(obj_path='/home/chase/Boyka/SCUT-FBP5500_v2/images',
                       label_path='/home/chase/Boyka/SCUT-FBP5500_v2/train_test_files/split64/test.txt',
                       data_transform=transform, target_transform=transform)

train_loader = data.DataLoader(dataset=train_set, batch_size=16, num_workers=8, shuffle=True)
test_loader = data.DataLoader(dataset=test_set, batch_size=16, num_workers=8, shuffle=False)

num_epochs = 150
num = 20
# net = STN3d()
net = PointNetReg()
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
