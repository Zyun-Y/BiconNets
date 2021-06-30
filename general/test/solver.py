"""
BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection
Ziyun Yang, Somayyeh Soltanian-Zadeh and Sina Farsiu
Codes from: https://github.com/Zyun-Y/BiconNets
Paper: https://arxiv.org/abs/2103.00334
"""

from random import shuffle
import sys
import datetime
import numpy as np
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import os
from tensorboardX import SummaryWriter
import torchvision.utils as utils
from skimage.io import imread, imsave
from utils_bicon import *

save = 'save'
if not os.path.exists(save):
    os.makedirs(save)


class Solver(object):


    def run(self, model, train_loader, val_loader, cfg, mode):
        csv = 'results_'+str(cfg.exp_id)+'.csv'
        with open(os.path.join(save, csv), 'w') as f:
            f.write('epoch, test_dice, test_dice_binary, mae_binary, maxf_binary, meanf_bi, S-measure_bi\n')

        if mode =='train':
            self.train(model, train_loader, val_loader,cfg)
        else:
            self.test_epoch(model,val_loader,0,cfg.exp_id)

    def train(self, model, train_loader, val_loader,cfg):


        exp_id = cfg.exp_id
        model.cuda()


        self.test_epoch(model,val_loader,-1,exp_id)
        
    def test_epoch(self,model,loader,epoch,exp_id):

        n=0
        model.eval()
        with torch.no_grad(): 
            for j_batch, test_data in enumerate(loader):
                curr_dice = []
                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])
                name = test_data[2]
                X_test= X_test.cuda()
                y_test = y_test.long().cuda()

                if X_test.shape[1]==1:
                    X_test = torch.cat([X_test,X_test,X_test],1)
                output_test = model(X_test)

                pred = bv_test(output_test)   
                for im in range(y_test.shape[0]):
                    save_fig(pred[im][0],y_test[im][0],exp_id,j_batch,im,epoch,name[im])

def save_fig(pred,y_test,exp_id,j_batch,n,epoch,name):
    path = 'output/'+str(exp_id)
    path1 = 'output/'+str(exp_id)+'/gt'
    path2 = 'output/'+str(exp_id)+'/pred'
    if not os.path.exists(path1):
        os.makedirs(path)
        os.makedirs(path1)
        os.makedirs(path2)
        # os.makedirs(path2)
        # os.makedirs(path2)
    # print('output/'+str(exp_id)+'/gt/'+name)
    
    imsave('output/'+str(exp_id)+'/gt/'+name,(y_test*255.0).cpu().numpy()
    imsave('output/'+str(exp_id)+'/pred/'+name,(pred*255.0).squeeze().cpu().numpy()
