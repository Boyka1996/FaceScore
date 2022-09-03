#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2022/9/3 23:46
@Author     : Boyka
@Contact    : zhw@s.upc.edu.cn
@File       : train_eval.py
@Project    : FaceScore
@Description:
"""
import copy
import csv
import torch.nn as nn
import numpy as np
import torch
import logging
import os
import datetime
from FaceData import FaceDataset
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from torchvision import models

logdir = './logs'
if not os.path.exists('./logs'):
    os.makedirs('./logs')
log_file_name = '%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
log_path = os.path.join(logdir, log_file_name + '.log')
logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)  # 将日志的输出级别调节为info
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M')
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(fmt=formatter)
file_handler = logging.FileHandler(filename=log_path, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(fmt=formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ******日志相关部分结束，前面的都不用动*****************


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet101(pretrained=True)
# 修改全连接层的输出
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
criterion = nn.MSELoss()
net = model
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 对图片尺寸做一个缩放切割
    transforms.RandomHorizontalFlip(),  # 水平翻转
    transforms.ToTensor(),  # 转化为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 进行归一化
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 进行归一化
])

batch_size = 16
num_workers = 8
num = 20
names = []
labels = []
label_path = ''
image_path = '/home/chase/Boyka/SCUT-FBP5500_v2/Images'

with open(label_path) as file:
    reader = csv.DictReader(file)
    for row in reader:
        names.append(row['id'])
        labels.append(row['UV'])

gt = np.array([])
pre = np.array([])
for round_idx in range(len(names)):
    train_names, train_labels = copy.deepcopy(names), copy.deepcopy(labels)
    del train_names[round_idx]
    del train_labels[round_idx]
    train_set = FaceDataset(img_path=image_path,
                            data=train_names,
                            label=train_labels,
                            transform=train_transforms)
    train_loader = data.DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    run_loss, total_loss = 0.0, 0.0
    run_ae, total_ae = 0.0, 0.0
    # 训练
    net.train()
    for i, data in enumerate(train_loader):
        obj, label = data
        obj, label = obj.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = net(obj)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        ae = torch.mean(torch.abs(outputs - label))
        run_ae += ae.item()
        if i % num == 0:
            logger.info(
                'train epoch %d, %2d loss: %.3f, train mae: %.3f' % (round_idx, i / num, run_loss / num, run_ae / num))
            run_loss = 0
            run_ae = 0
        total_loss += loss.item()
        total_ae += ae.item()
    logger.info("epoch %d train mae: %.3f " % (round_idx, total_ae / len(train_loader)))
    # 测试
    net.eval()
    img_path = os.path.join(image_path, names[round_idx])
    obj = Image.open(img_path)
    obj = test_transforms(obj)
    label = torch.tensor([labels[round_idx]])
    obj, label = obj.to(device), label.to(device)
    outputs = net(obj)
    gt = np.append(labels[round_idx])
    pre_value = outputs.cpu().detach().numpy().squeeze()
    pre = np.append(pre_value)
    logger.info("data: %d ground truth: %f, predict: %f " % (round_idx, labels[round_idx], pre_value))
logger.info("test corrcoef: %.3f ：" % (np.corrcoef(gt, pre)[0][1]))
if not os.path.exists('./models'):
    os.makedirs('./models')
torch.save(net.state_dict(), os.path.join('./models', log_file_name + '.pth'))

print("finished training!")
