import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.backends import cudnn
from model import build_model, weights_init
import scipy.misc as sm
import numpy as np
import os
from utils_bicon import *
import torchvision.utils as vutils
import cv2
from torch.optim import lr_scheduler
import torch.nn.functional as F
import math
from connect_loss import bicon_loss
import time
import sys
import PIL.Image
import scipy.io
import os
import logging
from apex import amp
EPSILON = 1e-8
p = OrderedDict()

from dataset import get_loader
base_model_cfg = 'resnet'
p['lr_bone'] = 2e-5  # Learning rate resnet:5e-5, vgg:2e-5
p['lr_branch'] = 0.025  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.90  # Momentum
lr_decay_epoch = [10,25] # [6, 9], now x3 #15
nAveGrad = 10  # Update the weights once in 'nAveGrad' forward passes
showEvery = 50
tmp_path = 'tmp_see'

save = 'save'
if not os.path.exists(save):
    os.makedirs(save)

class Solver(object):
    def __init__(self, train_loader, test_loader, config, save_fold=None):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.loss = bicon_loss()
        self.save_fold = save_fold
        self.mean = torch.Tensor([123.68, 116.779, 103.939]).view(3, 1, 1) / 255.
        # inference: choose the side map (see paper)
        if config.visdom:
            self.visual = Viz_visdom("trueUnify", 1)
        self.build_model()

        if self.config.pre_trained: self.net.load_state_dict(torch.load(self.config.pre_trained))
        if config.mode == 'train':
            self.log_output = open("%s/logs/log.txt" % config.save_fold, 'w')
        else:
            print('Loading pre-trained model from %s...' % self.config.model)
            self.net_bone.load_state_dict(torch.load(self.config.model))
            self.net_bone.eval()

    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def get_params(self, base_lr):
        ml = []
        for name, module in self.net_bone.named_children():
            print(name)
            if name == 'loss_weight':
                ml.append({'params': module.parameters(), 'lr': p['lr_branch']})          
            else:
                ml.append({'params': module.parameters()})
        return ml

    # build the network
    def build_model(self):
        self.net_bone = build_model(base_model_cfg)
        if self.config.cuda:
            self.net_bone = self.net_bone.cuda()
            
        self.net_bone.eval()  # use_global_stats = True
        self.net_bone.apply(weights_init)
        if self.config.mode == 'train':
            if self.config.load_bone == '':
                if base_model_cfg == 'vgg':
                    self.net_bone.base.load_pretrained_model(torch.load(self.config.vgg))
                elif base_model_cfg == 'resnet':
                    self.net_bone.base.load_state_dict(torch.load(self.config.resnet))
            if self.config.load_bone != '': self.net_bone.load_state_dict(torch.load(self.config.load_bone))

        self.lr_bone = p['lr_bone']
        self.lr_branch = p['lr_branch']
        self.optimizer_bone = Adam(filter(lambda p: p.requires_grad, self.net_bone.parameters()), lr=self.lr_bone, weight_decay=p['wd'])

        self.print_network(self.net_bone, 'trueUnify bone part')

    # update the learning rate
    def update_lr(self, rate):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * rate

    def Eval_mae(self,pred,gt):
        avg_mae, img_num = 0.0, 0.0
        #mae_list = [] # for debug
        with torch.no_grad():
            mea = torch.abs(pred - gt).mean()

        return mea.item()

    def test(self,epoch, test_mode=0):
        

        mae_ls_binary  = []
        fmax_ls_binary = []

        for batch_id, data_batch in enumerate(self.test_loader):
            images_,y_test, name, im_size = data_batch['image'],data_batch['label'], data_batch['name'][0], np.asarray(data_batch['size'])
            # print(i)
            with torch.no_grad():
                
                images = Variable(images_)
                if self.config.cuda:
                    images = images.cuda()
                    y_test = y_test.cuda()
                # print(images.size())
                time_start = time.time()
                _, _, up_sal_f = self.net_bone(images)
                output_test = up_sal_f[-1].cuda()
                torch.cuda.synchronize()
                time_end = time.time()

                pred=bv_test(output_test)

                # print(pred.shape, y_test.shape)
                mae_bi = self.Eval_mae(pred[0][0],y_test[0][0])
                mae_ls_binary.append(mae_bi)

                
                if batch_id%100 == 0:
                    print('test [Iteration : ' + str(batch_id) + '/' + str(len(self.test_loader)) + ']' ' ave mae f:%.3f' %(
                        np.mean(mae_ls_binary)))

        with open('save/results.csv', 'a') as f:
            f.write('%03d,%0.5f \n' % (
                (epoch + 1),
                np.mean(mae_ls_binary)))

        return(np.mean(mae_ls_binary))


   
    # training phase
    def train(self):
        scheduler = lr_scheduler.MultiStepLR(self.optimizer_bone, milestones=[15,25],
                                        gamma=0.1) 
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        F_v = 0
        best_mae = 1
        csv = 'results.csv'
        with open(os.path.join('save', csv), 'w') as f:
            f.write('epoch, mae_binary\n')

        self.net_bone, self.optimizer_bone = amp.initialize(self.net_bone, self.optimizer_bone, opt_level='O2')
        if not os.path.exists(tmp_path): 
            os.mkdir(tmp_path)

        self.test(0)

        for epoch in range(self.config.epoch):                          
            r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0
            self.net_bone.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label, sal_edge, conn = data_batch['sal_image'], data_batch['sal_label'], data_batch['sal_edge'], data_batch['sal_conn']
                if sal_image.size()[2:] != sal_label.size()[2:]:
                    print("Skip this batch")
                    continue
                sal_image, sal_label, sal_edge,  conn = Variable(sal_image), Variable(sal_label), Variable(sal_edge), Variable(conn)
                if self.config.cuda: 
                    sal_image, sal_label, sal_edge, conn = sal_image.cuda(), sal_label.cuda(), sal_edge.cuda(), conn.cuda()

                up_edge, up_sal, up_sal_f = self.net_bone(sal_image)

                loss_iter, r_edge_loss, r_sal_loss, r_sum_loss = self.loss(up_edge, up_sal, up_sal_f, sal_label,sal_edge, conn,None,True, False)
                loss = loss_iter / (nAveGrad * self.config.batch_size)

                with amp.scale_loss(loss, self.optimizer_bone) as scale_loss:
                    scale_loss.backward()
                aveGrad += 1

                if aveGrad % nAveGrad == 0:
       
                    self.optimizer_bone.step()
                    self.optimizer_bone.zero_grad()           
                    aveGrad = 0


                if i % showEvery == 0:

                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Edge : %10.4f  ||  Sal : %10.4f  ||  Sum : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num,  r_edge_loss*(nAveGrad * self.config.batch_size)/showEvery,
                                                                r_sal_loss*(nAveGrad * self.config.batch_size)/showEvery,
                                                                r_sum_loss*(nAveGrad * self.config.batch_size)/showEvery))

                    print('Learning rate: ' + str(self.lr_bone))
                    r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0

            
            current_mae = self.test(epoch)
            if current_mae < best_mae:
                best_mae = current_mae
                torch.save(self.net_bone.state_dict(), '%s/models/best_moel.pth' % (self.config.save_fold))

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net_bone.state_dict(), '%s/models/epoch_%d_bone.pth' % (self.config.save_fold, epoch + 1))
            scheduler.step()

        torch.save(self.net_bone.state_dict(), '%s/models/final_bone.pth' % self.config.save_fold)
        


