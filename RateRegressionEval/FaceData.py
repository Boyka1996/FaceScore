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
    def __init__(self, img_path, data, label, transform):
        self.img_path = img_path
        self.transform = transform
        self.data, self.label = data, label
        self.length = self.__len__()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_path, self.data[index])
        label = self.label[index]
        data = Image.open(img_path)
        data = self.transform(data)
        label = torch.from_numpy(np.ndarray(label))
        return data, label
