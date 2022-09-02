#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/9/2 22:29
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : FaceData.py
@Project    : FaceScore
@Description:
"""
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os


class FaceDataset(Dataset):
    def __init__(self, img_path, label_path, transform):
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform
        self.data, self.label = self.__build_dataset__()
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

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.data[index])
        label = self.label[index]
        data = Image.open(img_path)
        data = self.transform(data)
        label = torch.from_numpy(np.ndarray(label))
        return data, label
