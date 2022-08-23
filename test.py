#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/23 9:19
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : test.py
@Project    : FaceScore
@Description:
"""


# torch.save(model.state_dict(), PATH)
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
import os
import numpy as np
import torch

from torchvision import transforms

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
net = torch.load('')
net = net.to(device)
net.eval()
transform = transforms.Compose([
    transforms.ToTensor()
])


def predict(obj_path):
    obj = obj_transform(obj_path)
    obj = obj.to(device)
    outputs = net(obj)
    return outputs


def obj_transform(obj_path):
    if not os.path.exists(obj_path):
        return
    with open(obj_path) as file:
        v = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                v.append([float(strs[1]), float(strs[2]), float(strs[3])])
    return np.array(v)[0:-1:2].astype(np.float32)


if __name__ == '__main__':
    obj_folder = ''
    label_path = ''
    obj_name = ''
