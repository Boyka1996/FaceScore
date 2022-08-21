#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/21 11:21
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : model.py.py
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

#神经网络需要定义两个函数
#分别是构造函数，前向传播
#自定义的神经网络需要继承nn.Module
class Net(nn.Module):

    #构造函数
    def __init__(self):
        super(Net, self).__init__()
        #卷积层三个参数：in_channel, out_channels, 5*5 kernal
        self.con1 = nn.Conv2d(3, 10, 5)
        self.con2 = nn.Conv2d(10, 10, 5)
        #全连接层两个参数：in_channels, out_channels
        self.fc1 = nn.Linear(10 * 5 * 5, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)

    #前向传播
    def forward(self, input):
        #卷积 --> 激活函数（Relu) --> 池化
        x = self.con1(input)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        #重复上述过程
        x = self.con2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))

        #展平
        x = x.view(-1, self.num_flat_features(x))

        #全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


    #展平
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features = num_features * i
        return num_features

