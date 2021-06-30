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
from network.GCPA import GCPANet
from get_data import get_data, Config
import dataset

torch.cuda.set_device(1)

mode = 'norm'
for exp_id in range(6):

    if exp_id ==3:
        dataset = 'PASCAL'
    if exp_id ==1: 
        dataset = 'HKU'
    if exp_id ==2:
        dataset = 'DUT-O'
    if exp_id ==4:
        dataset = 'DUTS'
    if exp_id ==0:
        dataset = 'ECSSD'
    if exp_id ==5:
        dataset = 'sod'
    train_data, test_data = get_data(mode=mode,data=dataset)


    train_loader = torch.utils.data.DataLoader(train_data,pin_memory=(torch.cuda.is_available()), batch_size=16,shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(test_data,pin_memory=(torch.cuda.is_available()), batch_size=1, shuffle=False, num_workers=4)

    print("Train size: %i" % len(train_loader))
    print("Test size: %i" % len(val_loader))




    model = GCPANet(8).cuda()

    device = torch.device('cpu')
    model.load_state_dict(torch.load('/GCPANet_model.pth',map_location=device))
    
    cfg    = Config(lr=0.05, momen=0.9, decay=5e-4, epoch=32,exp_id= exp_id+1,betas=(0.9, 0.999),eps=1e-8)
    solver = Solver()

    solver.train(model, train_loader, val_loader,cfg)
