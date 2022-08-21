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
#神经网络
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import FaceDataset
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from model import Net
criterion = nn.MSELoss()

net = Net()

optimizer = optim.Adam(net.parameters(), lr = 0.001)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)
train_set=FaceDataset(obj_path='',label_path='')
test_set=FaceDataset(obj_path='',label_path='')


train_loader = Data.DataLoader(dataset = train_set, batch_size = 8, shuffle = True)
test_loader = Data.DataLoader(dataset = test_set, batch_size = 8, shuffle = False)

num_peochs = 1


#开始训练：num_peochs是训练周期数
for epoch in range(num_peochs):
    correct = 0
    total = 0
    run_loss = 0.0
    for i, data in enumerate(train_loader):
        input, label = data
        optimizer.zero_grad()
        outputs = net(input)
        lossValue = criterion(outputs, label)
        lossValue.backward()
        optimizer.step()
        run_loss += lossValue.item()

        num = 20

        if i % num == num - 1:
            print('[%d, %5d] loss : %.3f' % (epoch + 1, i + 1, run_loss / num))
            run_loss = 0

        _, pred = outputs.max(1)
        correct += (pred == label).sum().item()
        total += label.size()[0]

    print("训练集准确率：", correct / total)

#打印训练结束标识符
print("finished training!")
