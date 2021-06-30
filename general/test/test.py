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
#torch.set_default_tensor_type('torch.FloatTensor')

torch.cuda.set_device(0)

mode = 'norm_by_data'

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

    train_loader = torch.utils.data.DataLoader(train_data,pin_memory=(torch.cuda.is_available()), batch_size=1,shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(test_data,pin_memory=(torch.cuda.is_available()), batch_size=1, shuffle=False, num_workers=4)
    print("Train size: %i" % len(train_loader))
    print("Test size: %i" % len(val_loader))


    #your model###
    # model = build_model('resnet').cuda()



    device = torch.device('cpu')

    #load model
    # model.load_state_dict(torch.load('/epoch_30.pth',map_location=device))
    # model = model.cuda()
    cfg    = Config(lr=2e-6, momen=0.9, decay=5e-4, epoch=12,exp_id= exp_id+1,real_batch=10,betas=(0.9, 0.999),eps=1e-8)
    solver = Solver()

    solver.run(model, train_loader, val_loader,cfg,'test')
    
