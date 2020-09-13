# from torch.utils.data import Dataset
# class firstdataset(Dataset):
#     def __init__(self):
#         #初始化文件路径和文件名列表。在这里我们要做的工作就是初始化该类的一些基本参数
#         pass
#     def __getitem__(self, item):
#         #从文件中读取一个数据（例如，使用numpy.fromfile,PIL.Image.open）
#         #预处理数据（例如torchvision.transforms）
#         #返回数据对（例如图像和标签）
#         #这里需要注意的是，第一步read one data,是一个data
#         pass
#     def __len__(self):
#         #应该将0更改为数据集的总大小
#         pass
# 下面就来具体实现自定义数据集
import torch.nn.functional as F
import torch
import copy
import time
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms,utils,models,datasets
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import json
import matplotlib.pyplot as plt

class LandslideDataset(Dataset):
    def __init__(self,slide_dir,transforms=None,shuffle=True):
        self.root_dir=slide_dir#文件目录
        self.img_root=os.path.join(self.root_dir,)

    def __getitem__(self, index):#根据索引index获得该图片
    def __len__(self):

