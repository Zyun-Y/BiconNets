"""
BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection
Ziyun Yang, Somayyeh Soltanian-Zadeh and Sina Farsiu
Codes from: https://github.com/Zyun-Y/BiconNets
Paper: https://arxiv.org/abs/2103.00334
"""

import os
import cv2
import torch
import numpy as np
import glob
import torch.nn.functional as F
from dataset import Data

## put your data here ###
root = '/hpc/home/zy104/storage/SOD/DUTS-TR'
ECSSD = '/hpc/home/zy104/storage/SOD/ECSSD'
pascal = '/hpc/home/zy104/storage/SOD/PASCALS'
omron = '/hpc/home/zy104/storage/SOD/DUT-O'
hku = '/hpc/home/zy104/storage/SOD/HKU-IS'
duts = '/hpc/home/zy104/storage/SOD/DUTS-TE'

data_dict = {'train':root, 'ECSSD':ECSSD, 'PASCAL': pascal, 'HKU':hku, 'DUTS':duts,'DUT-O':omron}

class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs


        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None

def get_data(config,mode,data):

	cfg_test = Config(datapath=data_dict[data],mode=data, size = config.size)
	data_test = Data(cfg_test,mode)

	cfg    = Config(datapath=data_dict['train'], mode='train', size = config.size)
	data_train   = Data(cfg,mode)
	# print(cfg.datapath)
	# print(data)
	# print(data_dict[data])
	# cfg_test = Config(datapath=data_dict[data],mode=data)
	# data_test = Data(cfg_test,mode)

	# if mode == 'orginal':


	return data_train, data_test

