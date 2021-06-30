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


torch.cuda.set_device(0)

mode = 'norm_by_data'


for exp_id in range(1):

    dataset = 'ECSSD'

    train_data, test_data = get_data(mode=mode,data=dataset)

    train_loader = torch.utils.data.DataLoader(train_data,pin_memory=(torch.cuda.is_available()), batch_size=1,shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(test_data,pin_memory=(torch.cuda.is_available()), batch_size=1, shuffle=False, num_workers=4)

    print("Train size: %i" % len(train_loader))
    print("Test size: %i" % len(val_loader))


    ### your network
    # model = PoolNet()

    
    #your hyperparameter
    cfg    = Config(lr=2e-5, momen=0.9, decay=5e-4, epoch=35,exp_id= exp_id+1,real_batch=10, betas=(0.9, 0.999),eps=1e-8)
    solver = Solver()

    solver.train(model, train_loader, val_loader,cfg)


