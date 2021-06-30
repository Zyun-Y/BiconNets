import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

import glob
from torchvision import datasets, transforms
from skimage.io import imread, imsave
from solver import Solver
import torch.nn.functional as F
import torch.nn as nn

from network.poolnet import PoolNet, build_model,weights_init
from get_data import get_data, Config
import dataset


torch.cuda.set_device(1)

mode = 'norm_by_data'

def my_collate(batch):
    # print(len(batch))

    
    batch.sort(key=lambda x:x[1].shape[2],reverse=True)
    w=batch[0][1].shape[2]
    batch.sort(key=lambda x:x[1].shape[1],reverse=True)
    h=batch[0][1].shape[1]
    # print(h,w)
    c = len(batch)

    data0 = torch.zeros([c,3,h,w])
    gt0 = torch.zeros([c,1,h,w])
    conn0 = torch.zeros([c,8,h,w])
    i = 0
    for item in batch:
        # print(item[0].shape)
        hh = item[0].shape[1]
        ww = item[0].shape[2]
        data0[i,:3,:hh,:ww] = item[0]
        gt0[i,0,:hh,:ww] = item[1]
        conn0[i,:8,:hh,:ww] = item[2]
        i=i+1
    # target = torch.LongTensor(target)
    return [data0,gt0,conn0]


for exp_id in range(1):

    if exp_id ==0:
        dataset = 'ECSSD'

    train_data, test_data = get_data(mode=mode,data=dataset)



    train_loader = torch.utils.data.DataLoader(train_data,pin_memory=(torch.cuda.is_available()), batch_size=1,shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(test_data,pin_memory=(torch.cuda.is_available()), batch_size=1, shuffle=False, num_workers=4)

    print("Train size: %i" % len(train_loader))
    print("Test size: %i" % len(val_loader))



    model = build_model('resnet').cuda()
    model.apply(weights_init)
    model.base.load_pretrained_model('/github/resnet50-19c8e357.pth')

    
    print(model)
    device = torch.device('cpu')

    cfg    = Config(lr=2e-5, momen=0.9, decay=5e-4, epoch=30,exp_id= exp_id+1,real_batch=10, betas=(0.9, 0.999),eps=1e-8)
    solver = Solver()

    solver.train(model, train_loader, val_loader,cfg)


