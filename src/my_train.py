import numpy as np
import cv2
import os
import random
import pandas as pd
import torch

##############开始复现
from src.crowd_count import CrowdCount
from src.data_multithread_preload import multithread_dataloader
from src.models import Model
from src.data_multithread_preload import PreloadData
import  torchvision

image_path=r"F:\CV project\shanghaitech\part_B_final\train_data\images"
density_map_path=r"F:\CV project\shanghaitech\part_B_final\train_data\ground_truth"


net=CrowdCount()
vgg16=torch.load("VGG16.pth")
prior=net.features.prior
vgg=net.features.vgg16


vgg_16=[]
for k,v in  vgg16.state_dict().items():
    vgg_16.append(v)
useful_vgg16=vgg_16[:26]


i=0
useful_dict={}

for k,v in prior.state_dict().items():
    if(i<26):
        useful_dict[k]=useful_vgg16[i]
        i=i+1


prior_dict=prior.state_dict()
vgg_dict=vgg.state_dict()
prior_dict.update(useful_dict)
vgg_dict.update(useful_dict)



train_flag = dict()
train_flag['preload'] = False
train_flag['label'] = False
train_flag['mask'] = False

original_dataset_name = 'shanghaitechB_train'
train_data_config = dict()
train_data_config['shanghaitechB_train'] = train_flag.copy()
all_data = multithread_dataloader(train_data_config)

my_loss

