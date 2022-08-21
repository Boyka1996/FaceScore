#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/21 11:13
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : test.py
@Project    : FaceScore
@Description:
"""
import os
import numpy as np
"""
有用的其实只要定点v，因为vt纹理坐标和vn顶点法向量都是0，
f面为v/vt/vn v/vt/vn v/vt/vn（f 顶点索引 / 纹理坐标索引 / 顶点法向量索引）也没什么意义，
原来没有颜色是因为没有将原图的颜色映射回去
"""
def get_data(objFilePath):
    with open(objFilePath) as file:
        v = []
        vt = []
        vn = []
        f = []

        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                v.append([float(strs[1]), float(strs[2]), float(strs[3])])
            elif strs[0] == "vt":
                vt.append([float(strs[1]), float(strs[2]), float(strs[3])])
            elif strs[0] == "vn":
                vn.append([float(strs[1]), float(strs[2]), float(strs[3])])
            elif strs[0] == "f":
                f.append([float(strs[1]), float(strs[2]), float(strs[3])])
    return v,vt,vn,f