#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/8/23 12:22
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : temp.py
@Project    : FaceScore
@Description:
"""
import numpy as np
#
# x=np.array([1,3,5,2,3,5])
# y=np.array([1,3,4,5,6,3])
x=np.array([1,3,5])
y=np.array([1,3,4])
pc=np.corrcoef(x,y)

print(pc)


import numpy as np

x=np.array([1,3,5])
y=np.array([1,3,4])
n=len(x)

sum_xy = np.sum(np.sum(x*y))
sum_x = np.sum(np.sum(x))
sum_y = np.sum(np.sum(y))
sum_x2 = np.sum(np.sum(x*x))
sum_y2 = np.sum(np.sum(y*y))
pc = (n*sum_xy-sum_x*sum_y)/np.sqrt((n*sum_x2-sum_x*sum_x)*(n*sum_y2-sum_y*sum_y))

print(pc)
