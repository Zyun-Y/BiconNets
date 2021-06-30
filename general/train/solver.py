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
from connect_loss import bicon_loss
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import os
from tensorboardX import SummaryWriter
import torchvision.utils as utils
from skimage.io import imread, imsave
from lib.bad_grad_viz import register_hooks
from utils_bicon import *
from lr_update import get_lr

save = 'save'
if not os.path.exists(save):
    os.makedirs(save)

def create_exp_directory(exp_id):
    if not os.path.exists('models/' + str(exp_id)):
        os.makedirs('models/' + str(exp_id))


class Solver(object):

    gamma = 0.1
    step_size = 15


    def train(self, model, train_loader, val_loader,cfg):

        # replace your loss function with a bicon loss here
        self.loss_func = bicon_loss()

        num_epochs = cfg.epoch
        exp_id = cfg.exp_id
        opt_group = 1
        real_batch = cfg.real_batch
        lr_mode = 'warm-up-step'
        if opt_group == 2:
            base, head = [], []
            for name, param in model.named_parameters():
                if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
                    print(name)
                elif 'bkbone' in name:
                    base.append(param)
                else:
                    head.append(param)
            optim = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
        else:
            optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay,betas=cfg.betas)
    
        
            scheduler = lr_scheduler.StepLR(optim, step_size=self.step_size,gamma=self.gamma) 

        # model, optim = amp.initialize(model, optim, opt_level='O2')
        model.cuda()

        print('START TRAIN.')
        curr_iter = 0

        create_exp_directory(exp_id)
        csv = 'results_'+str(exp_id)+'.csv'
        with open(os.path.join(save, csv), 'w') as f:
            f.write('epoch, mae \n')

        best_mae = 0
        step=0
        m=0
        db_size = len(train_loader)/real_batch

        model.eval()
        for epoch in range(num_epochs):
            total_loss = 0
            model.zero_grad()
            iter_num = 0
            curr_step = 0


            for i_batch, sample_batched in enumerate(train_loader):
                step +=1
                curr_step += 1
                
                X = Variable(sample_batched[0])
                y = Variable(sample_batched[1])
                connect = Variable(sample_batched[2])

                X= X.cuda()
                y = y.long().cuda()
                connect = connect.cuda()
                

                out = model(X)

                loss_iter = self.loss_func(out, y,connect)

                loss = loss_iter/real_batch
                total_loss += loss.item()
                iter_num +=1

                loss.backward()

                # Update gradient every real_batch batch. Use this part if you want to train your network with the original image size. 
                # Otherwise just ignore this and use the normal way to update gradient.
                if iter_num%real_batch ==0:

                    m +=1
                    optim.step()
                    optim.zero_grad()
                    iter_num=0

                    if m == 10:
                        print('%s | step:%d/%d/%d/%d | lr=%.6f | loss=%.6f'%(datetime.datetime.now(), step/real_batch,curr_step/real_batch, epoch+1,num_epochs, optim.param_groups[0]['lr'], total_loss))
                        m=0
                    total_loss=0

            scheduler.step()

            curr_mae = self.test_epoch(model,val_loader,epoch,exp_id)
            if curr_mae<best_mae:
                best_mae = curr_mae
                torch.save(model.state_dict(), 'models/' + str(exp_id) + '/'+'best_model.pth')
            if epoch>15:
                torch.save(model.state_dict(), 'models/' + str(exp_id) + '/relaynet_epoch' + str(epoch + 1)+'.pth')

        print('FINISH.')
        
    def test_epoch(self,model,loader,epoch,exp_id):
        mae_ls_binary = []

        model.eval()
        with torch.no_grad(): 
            for j_batch, test_data in enumerate(loader):
                curr_dice = []
                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])

                X_test= X_test.cuda()

                y_test = y_test.long().cuda()

                output_test = model(X_test)

                pred = bv_test(output_test)

                for im in range(y_test.shape[0]):

                    mae_bi = self.Eval_mae(pred[im],y_test[im])
                    mae_ls_binary.append(mae_bi)

                if j_batch % 100 ==0:
                    print('test [Iteration : ' + str(j_batch) + '/' + str(len(loader)) + ']' ' ave mae:%.3f' %(np.mean(mae_ls_binary)))

            csv = 'results_'+str(exp_id)+'.csv'

            with open(os.path.join(save, csv), 'a') as f:
                f.write('%03d,%0.6f\n' % (
                    (epoch + 1),
                    np.mean(mae_ls_binary)))


            return np.mean(mae_ls_binary)


    def Eval_mae(self,pred,gt):
        avg_mae, img_num = 0.0, 0.0
        #mae_list = [] # for debug
        with torch.no_grad():
            mea = torch.abs(pred - gt).mean()

        return mea.item()
