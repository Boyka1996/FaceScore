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
# 神经网络
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.autograd import Variable


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.fc4=nn.Linear(9, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        print(x.shape)
        batchsize = x.size()[0] # shape (batch_size,3,point_nums)
        x = F.relu(self.bn1(self.conv1(x))) # shape (batch_size,64,point_nums)
        x = F.relu(self.bn2(self.conv2(x))) # shape (batch_size,128,point_nums)
        x = F.relu(self.bn3(self.conv3(x))) # shape (batch_size,1024,point_nums)
        x = torch.max(x, 2, keepdim=True)[0] # shape (batch_size,1024,1)
        x = x.view(-1, 1024) # shape (batch_size,1024)

        x = F.relu(self.bn4(self.fc1(x))) # shape (batch_size,512)
        x = F.relu(self.bn5(self.fc2(x))) # shape (batch_size,256)
        x = self.fc3(x) # shape (batch_size,9)
        x = F.relu(x)
        x=self.fc4(x)
        return x
        #
        # iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
        #     batchsize, 1) # # shape (batch_size,9)
        # if x.is_cuda:
        #     iden = iden.cuda()
        # # that's the same thing as adding a diagonal matrix(full 1)
        # x = x + iden # iden means that add the input-self
        # x = x.view(-1, 3, 3) # shape (batch_size,3,3)
        # return x

if __name__ == '__main__':
    device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    sim_data = torch.rand(32, 3, 20000)
    print(sim_data.shape)
    # sim_data=sim_data.cuda(1)
    sim_data=sim_data.to(device)
    trans = STN3d()
    # trans=trans.cuda(1)
    trans=trans.to(device)
    start_time=time.time()
    out = trans(sim_data)
    # print(out)
    print(time.time()-start_time)
    # print('stn', out.size())
    # print('loss', feature_transform_regularizer(out))
