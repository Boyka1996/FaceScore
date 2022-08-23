#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/23 9:45
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : analysis.py
@Project    : FaceScore
@Description:
"""
import os
import numpy as np


def obj_transform(file_name):
    with open(file_name) as file:
        v = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                v.append([float(strs[1]), float(strs[2]), float(strs[3])])
    return np.array(v)


if __name__ == '__main__':
    obj_path = '/home/chase/Boyka/SCUT-FBP5500_v2/images'
    max_num = -1
    min_num = 1e5
    for obj_name in os.listdir(obj_path):
        obj_array = obj_transform(os.path.join(obj_path, obj_name))
        temp_max = np.max(obj_array)
        temp_min = np.min(obj_array)
        if temp_min < min_num:
            min_num = temp_min
        if temp_max > max_num:
            max_num = temp_max
        print(min_num, max_num)
