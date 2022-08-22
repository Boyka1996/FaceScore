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
import torchvision.transforms as transforms
import torch.utils.data as data
from model import Net

criterion = nn.MSELoss()

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
train_set = FaceDataset(obj_path='', label_path='/home/chase/Boyka/SCUT-FBP5500_v2/train_test_files/split64/train.txt')
test_set = FaceDataset(obj_path='', label_path='/home/chase/Boyka/SCUT-FBP5500_v2/train_test_files/split64/test.txt')

train_loader = data.DataLoader(dataset=train_set, batch_size=8, shuffle=True)
test_loader = data.DataLoader(dataset=test_set, batch_size=8, shuffle=False)

num_epochs = 1

for epoch in range(num_epochs):
    correct = 0
    run_loss = 0
    total_loss = 0.0
    for i, data in enumerate(train_loader):
        obj, label = data
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
