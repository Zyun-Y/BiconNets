import os
import cv2
import torch
import numpy as np
import glob
import torch.nn.functional as F
from dataset import Data

## put your data here ###
root = '/data/DUTS/DUTS-TR'
ECSSD = '/data/ECSSD'
pascal = '/data/PASCAL'
omron = '/data/DUTS/DUT-OMRON'
hku = '//data/HKU-IS'
duts = '/data/DUTS/DUTS-TE'

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

def get_data(mode,data):

	cfg_test = Config(datapath=data_dict[data],mode=data)
	data_test = Data(cfg_test,mode)

	cfg    = Config(datapath=data_dict['train'], mode='train')
	data_train   = Data(cfg,mode)
	# print(cfg.datapath)
	# print(data)
	# print(data_dict[data])
	# cfg_test = Config(datapath=data_dict[data],mode=data)
	# data_test = Data(cfg_test,mode)

	# if mode == 'orginal':


	return data_train, data_test

