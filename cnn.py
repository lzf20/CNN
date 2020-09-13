import torch
import torchvision
from torch import nn
from torchvision import transforms,models,datasets
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import copy
import json
import imageio
import warnings
import random
import sys
from PIL import Image

# 加载数据
data_dir="./data"
train_dir=data_dir+"train.txt"
valid_dir=data_dir+"val.txt"








