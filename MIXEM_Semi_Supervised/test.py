import sys, os,argparse, pickle,time
import torch
torch.manual_seed(179510)
import numpy as np
np.random.seed(179510)
import yaml
from sklearn import preprocessing
import importlib.util

from tqdm import tqdm
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

import os
from PIL import Image
import matplotlib.pyplot as plt


class NewDataset(torch.utils.data.Dataset):

    def __init__(self, path, mode= 1): #1 - supervised

        self.mode = mode
        self.path = path
        imgs = os.listdir(path)

        X = []

        if mode == 1:
            Y = []

            for t in imgs:
                X.append(t)
                Y.append(int(t[:-5][-1]))

            self.X = X
            self.Y = Y
            self.length = len(X)

        else:

            for t in imgs:
                X.append(t)
            
            self.length = len(X)
            self.X = X
    
    def __len__(self):
        return self.length 

    def __getitem__(self, ind):

        img_path = self.path + self.X[ind]
        img = Image.open(img_path)
        img = transforms.ToTensor()(img)

        if self.mode == 1:
            img_label = self.Y[ind]
            return img, img_label

        else:
            return img


train_path = '/content/drive/MyDrive/Fall 2021/18786 Introduction to Deep Learning/Project/Naaye2/dirty naai/'
train_data = NewDataset(train_path, mode= 1)

train_loader = DataLoader(train_data,
                          batch_size= 1, 
                          num_workers= 4,
                          shuffle= True)
print("got loader da naaye")

for data in train_loader:

    x, y = data
    print(x.shape, y.shape)
    break

