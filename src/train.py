import numpy as np
import cv2
import os
import random
import pandas as pd
import torch

##############开始复现
from src.crowd_count import CrowdCount
from src.data_multithread_preload import Data
from src.data_multithread_preload import PreloadData
from src.models import Model
from src.data_multithread_preload import PreloadData
from src.data_multithread_preload import multithread_dataloader
import  torchvision

image_path=r"F:\CV project\shanghaitech\part_B_final\train_data\images"
density_map_path=r"F:\CV project\shanghaitech\part_B_final\train_data\ground_truth"


################################## 作者这个vgg16完全坑人
################################## 还是要仔细看论文，我对作者直接命名为vgg16感到无语
# net=CrowdCount()
# vgg16=torch.load("VGG16.pth")
# prior=net.features.prior
# vgg=net.features.vgg16


########三个网络各自的结构
#####################################有一个很关键的问题，参数字典的键值对不上
#####################################手动初始化,筛选出原生vgg16适合的参数

#for k,v in  vgg16.state_dict().items():
#print(k,v.size())
##################保留了有效的参数

# vgg_16=[]
# for k,v in  vgg16.state_dict().items():
#     vgg_16.append(v)
# useful_vgg16=vgg_16[:26]
# # for v in useful_vgg16:
# #     print(v.size())
# # print(len(useful_vgg16))
#
#
# #######useful_dict是有用键值对的字典，建值取得和k相同
#
# i=0
# useful_dict={}
#
# for k,v in prior.state_dict().items():
#     if(i<26):
#         useful_dict[k]=useful_vgg16[i]
#         i=i+1
#
# # for k,v in  useful_dict.items():
# #     print(k,v.size())
#
# #############################加载两个网络的字典
# prior_dict=prior.state_dict()
# vgg_dict=vgg.state_dict()
#
# prior_dict.update(useful_dict)
# vgg_dict.update(useful_dict)

##############################手动初始化
########################检查是否初始化成功
###########################################顺便查看prior以及vgg参数
#################################检查初始化

# params_dict=[]
# i=0
# for k,v in net.features.prior.state_dict().items():
#     if(i<25):
#         params_dict.append(v)
#         i=i+1
# print(len(params_dict))
# print(len(useful_vgg16))

# for k,v in  useful_dict.items():
#     if (v==net.features.prior.state_dict()[k]).all():
#         print(k+"  is perfectly ininitialized")
#     else:
#         print("The prior was not innitialized well")

# for k,v in  useful_dict.items():
#     if (v==net.features.vgg16.state_dict()[k]).all():
#         print(k+"  is perfectly ininitialized")
#     else:
#         print("The vgg16"
#               " was not innitialized well")

# print(net.features.prior.state_dict()["12.conv.bias"])
# print(useful_dict["12.conv.bias"])


################################现在来弄弄这个特别的要死的数据加载类
################################很坑啊，readme有和没有有什么区别
################################这部分应该才是最难的
################################仿照test使用案例

# image_path=r"F:\CV project\shanghaitech\part_B_final\train_data\images"
# density_map_path=r"F:\CV project\shanghaitech\part_B_final\train_data\ground_truth"
#
# preload_data=PreloadData(image_path=image_path,density_map_path=density_map_path)
# data_loader=Data(preload_data)

train_flag = dict()
train_flag['preload'] = False
train_flag['label'] = False
train_flag['mask'] = False


original_dataset_name = 'shanghaitechB_train'
train_data_config = dict()
train_data_config['shanghaitechB_train'] = train_flag.copy()
all_data = multithread_dataloader(train_data_config)


###########################加载网络
##########################固定初始化部分参数

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

net.cuda()
net.train()

#############################训练参数
start_step = 0
end_step = 2000
lr = 0.00001
momentum = 0.9
disp_interval = 500
log_interval = 250


###########################优化器
################################存在初始化问题
params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)


####################开始训练
if __name__ == '__main__':
    for epoch in range(start_step, end_step+1):
        step = -1
        train_loss = 0
        for data_name in train_data_config:
            data = all_data[data_name]['data']
            for blob in data:
                image_data = blob['image']
                ground_truth_data = blob['density']
                roi = blob['roi']
                image_name = blob['image_name'][0]
                estimate_map, loss_dict, visual_dict = net(im_data=image_data, roi=roi, ground_truth=ground_truth_data)
                loss = loss_dict["total"]
                print(loss_dict["total"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
