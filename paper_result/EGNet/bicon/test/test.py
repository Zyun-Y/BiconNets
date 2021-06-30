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
from network.egnet import build_model, weights_init
from get_data import  Config
from dataset import get_loader

torch.cuda.set_device(1)

mode = 'norm_by_data'

### your datapath ###
root = '/data/DUTS/DUTS-TR'
ECSSD = '/data/ECSSD'
pascal = '/data/PASCAL'
omron = '/data/DUTS/DUT-OMRON'
hku = '/data/HKU-IS'
duts = '/data/DUTS/DUTS-TE'



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


    val_loader, dataset = get_loader(dataset,batch_size=1, mode='test')

    print("Test size: %i" % len(val_loader))


    model = build_model(base_model_cfg='resnet').cuda()
    model.eval() 
    model.apply(weights_init)


    model.base.load_state_dict(torch.load('/code/github/resnet50_caffe.pth'))
    device = torch.device('cpu')

    model.load_state_dict(torch.load('/EGNet_model',map_location=device))

    cfg    = Config(lr=0.000015, momen=0.9, decay=5e-4, epoch=32,exp_id= exp_id+1,betas=(0.9, 0.999),eps=1e-8)
    solver = Solver()

    solver.train(model, val_loader, val_loader,cfg)
