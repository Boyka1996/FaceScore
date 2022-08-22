#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/21 11:33
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : data.py
@Project    : FaceScore
@Description:
"""
import torch
import torch.utils.data as data
import numpy as np
import os


class FaceDataset(data.Dataset):
    def __init__(self, obj_path, label_path, data_transform, target_transform):
        self.obj_path = obj_path
        self.label_path = label_path
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.data, self.target = self.__build_dataset__()
        self.length = self.__len__()

    def __build_dataset__(self):
        names = []
        labels = []
        with open(self.label_path) as file:
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                names.append(strs[0])
                labels.append(float(strs[1]))
        return names, labels

    def __getitem__(self, index):
        obj, target = self.__obj_transform__(self.data[index]), np.array([self.target[index]])
        if self.data_transform:
            obj = self.data_transform(obj)
            obj = obj.squeeze().transpose(0, 1)
        if self.target_transform:
            target = torch.Tensor(target)
            # target = self.target_transform(target)
        return obj, target

    def __len__(self):
        return len(self.target)

    def __obj_transform__(self, file_name):
        obj_file_path = os.path.join(self.obj_path, file_name).replace('.jpg', '.obj')
        if not os.path.exists(obj_file_path):
            return
        with open(obj_file_path) as file:
            v = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    v.append([float(strs[1]), float(strs[2]), float(strs[3])])
        return np.array(v)[0:-1:2].astype(np.float32)
