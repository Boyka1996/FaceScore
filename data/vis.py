#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/22 10:18
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : vis.py
@Project    : FaceScore
@Description:
"""

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
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


def analysis(obj):
    temp = 0
    ana=[]
    for idx, data in enumerate(obj):
        if temp > data[2]:
            # print(idx)
            ana.append(idx)
        temp = data[2]
    print(len(ana))

obj_array = obj_transform('test/AF1.obj')
analysis(obj_array)
print(len(obj_array))
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(obj_array[:, 0], obj_array[:, 1], obj_array[:, 2], c=obj_array[:, 0], cmap='Greens')
plt.show()
