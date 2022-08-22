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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import FaceDataset
import torchvision
import logging
import os
import torchvision.transforms as transforms
import torch.utils.data as data
from model import STN3d
import datetime

argument_path = 'experiment_%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)
device = torch.device(device)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logdir = 'logs'
log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
log_path = log_file_name + '.log'
logging.basicConfig(
    filename=os.path.join(logdir, log_path),
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info(device)

criterion = nn.MSELoss()

net = STN3d()
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)
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

train_loader = data.DataLoader(dataset=train_set, batch_size=16, shuffle=True)
test_loader = data.DataLoader(dataset=test_set, batch_size=16, shuffle=False)

num_epochs = 1

for epoch in range(num_epochs):
    correct = 0
    run_loss = 0
    total_loss = 0.0
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

        num = 20

        if i % num == num - 1:
            print('[%d, %5d] loss : %.3f' % (epoch + 1, i + 1, run_loss / num))
            run_loss = 0

        _, pred = outputs.max(1)

        total_loss += loss.item()

    print("训练集误差：", total_loss / train_set.length)

print("finished training!")
