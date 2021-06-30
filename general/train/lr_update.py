"""
BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection
Ziyun Yang, Somayyeh Soltanian-Zadeh and Sina Farsiu
Codes from: https://github.com/Zyun-Y/BiconNets
Paper: https://arxiv.org/abs/2103.00334
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dataset
import os
import numpy as np


def get_lr(mode, epoch, epoch_num, step,db_size):
    if mode == 'warm-up-epoch':
        MAX_LR = 0.03
        lr = (1-abs((epoch+1)/(epoch_num+1)*2-1))*MAX_LR
        return lr

    if mode == 'warm-up-step':
        BASE_LR = 2e-5
        MAX_LR = 0.012

        niter = epoch * db_size + step
        lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, epoch_num*db_size, niter, ratio=1.)
        return lr,momentum

    if mode == 'custom':
        # step_size = 20
        # ratio = 0.1
        orig_lr = 2e-5
        lr = orig_lr
        if epoch == 15:
        	lr = orig_lr*0.1

        return lr



def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum
