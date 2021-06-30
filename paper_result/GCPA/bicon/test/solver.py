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
from lr_update import get_lr

save = 'save'
if not os.path.exists(save):
    os.makedirs(save)


class Solver(object):

    def train(self, model, train_loader, val_loader,cfg):
        exp_id = cfg.exp_id
        model.cuda()

        curr_iter = 0
        csv = 'results_'+str(exp_id)+'.csv'
        with open(os.path.join(save, csv), 'w') as f:
            f.write('epoch, MAE \n')

        self.test_epoch(model,val_loader,-1,exp_id)

    def test_epoch(self,model,loader,epoch,exp_id):
        step=0
        model.eval()
        with torch.no_grad(): 
            for j_batch, test_data in enumerate(loader):
                curr_dice = []
                step+=1
                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])

                name = test_data[2]
                X_test= X_test.cuda()

                y_test = y_test
                size = test_data[3]

                output_test,_,_,_ = model(X_test)

                pred = bv_test(output_test)

                y_test = torch.nn.functional.interpolate(y_test, size=size,mode='bilinear')
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

    y_test = (y_test*255.0).cpu().numpy()
    pred = (pred*255.0).squeeze().cpu().numpy()
    imsave('output/'+str(exp_id)+'/gt/'+name,y_test)
    imsave('output/'+str(exp_id)+'/pred/'+name,pred)