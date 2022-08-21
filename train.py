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
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from model import Net
criterion = nn.CrossEntropyLoss()

net = Net()

criterion = nn.CrossEntropyLoss()
#定义优化器种类：Adam
optimizer = optim.Adam(net.parameters(), lr = 0.001)
#定义变换：转化成Tensor。必须转化成张量形式才能求导。
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

#用于训练的训练集和测试集
transet = torchvision.datasets.CIFAR10(root = './data', train = True, download=True, transform = transform)
testset = torchvision.datasets.CIFAR10(root = './data', train=False, download=True, transform = transform)

trainLoader = Data.DataLoader(dataset = transet, batch_size = 40, shuffle = True)
testLoader = Data.DataLoader(dataset = testset, batch_size = 40, shuffle = False)

num_peochs = 1


#开始训练：num_peochs是训练周期数
for epoch in range(num_peochs):
    correct = 0
    total = 0
    run_loss = 0.0
    for i, data in enumerate(trainLoader):
        input, label = data
        input, label = input, label
        #梯度清零
        optimizer.zero_grad()
        #前向传播：计算输出
        outputs = net(input)
        #计算损失函数
        lossValue = criterion(outputs, label)
        #反向传播
        lossValue.backward()
        #参数更新
        optimizer.step()

        run_loss += lossValue.item()

        num = 20

        if i % num == num - 1:
            print('[%d, %5d] loss : %.3f' % (epoch + 1, i + 1, run_loss / num))
            run_loss = 0

        #训练集准确率
        _, pred = outputs.max(1)
        correct += (pred == label).sum().item()
        total += label.size()[0]

    print("训练集准确率：", correct / total)

#打印训练结束标识符
print("finished training!")
