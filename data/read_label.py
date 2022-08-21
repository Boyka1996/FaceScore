#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/21 17:41
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : read_label.py
@Project    : FaceScore
@Description:
"""
file_path = 'test.txt'
names = []
labels = []
with open(file_path) as file:
    while 1:
        line = file.readline()
        if not line:
            break
        strs = line.split(" ")
        names.append(strs[0])
        labels.append(float(strs[1]))

print(names)
