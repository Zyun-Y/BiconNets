"""
BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection
Ziyun Yang, Somayyeh Soltanian-Zadeh and Sina Farsiu
Codes from: https://github.com/Zyun-Y/BiconNets
Paper: https://arxiv.org/abs/2103.00334
"""

import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import scipy.io as scio

Directable={'upper_left':[-1,-1],'up':[0,-1],'upper_right':[1,-1],'left':[-1,0],'right':[1,0],'lower_left':[-1,1],'down':[0,1],'lower_right':[1,1]}
TL_table = ['lower_right','down','lower_left','right','left','upper_right','up','upper_left']


# hori_translation = torch.zeros([256,400])
# for i in range(399):
#     hori_translation[i,i+1] = torch.tensor(1.0)
# verti_translation = torch.zeros([300,300])
# for j in range(299):
#     verti_translation[j,j+1] = torch.tensor(1.0)
# hori_translation = hori_translation.double()
# verti_translation = verti_translation.double()




def edge_loss(glo_map,vote_out,edge,target):
    pred_mask_min, _ = torch.min(vote_out, dim=1)
    pred_mask_max,_ = torch.max(vote_out, dim=1)
    pred_mask_min = 1-pred_mask_min
    pred_mask_min = pred_mask_min * edge
    decouple_map = glo_map*(1-edge)+pred_mask_min

    minloss = F.binary_cross_entropy(decouple_map.unsqueeze(1),target,reduction='sum')
    return minloss

class bicon_loss(nn.Module):
    def __init__(self):
        super(bicon_loss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss(reduction='sum')

        hori_translation = torch.zeros([1,num_class, c_map.shape[3],c_map.shape[3]])
        for i in range(c_map.shape[3]-1):
            hori_translation[:,:,i,i+1] = torch.tensor(1.0)
        verti_translation = torch.zeros([1,num_class,c_map.shape[2],c_map.shape[2]])
        for j in range(c_map.shape[2]-1):
            verti_translation[:,:,j,j+1] = torch.tensor(1.0)
        self.hori_trans = hori_translation.float().cuda()
        self.verti_trans = verti_translation.float().cuda()

        
    def forward(self, c_map, target, con_target):
        num_class = 1
        batch_num = c_map.shape[0]

        con_target = con_target.type(torch.FloatTensor).cuda()

        #find edge ground truth
        sum_conn = torch.sum(con_target,dim=1)
        edge = torch.where(sum_conn<8,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        edge0 = torch.where(sum_conn>0,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        edge = edge*edge0


        target = target.type(torch.FloatTensor).cuda()
        c_map = F.sigmoid(c_map)
        
        # construct the translation matrix
        self.hori_translation = self.hori_trans.repeat(batch_num,1,1,1).cuda()
        self.verti_translation  = self.verti_trans.repeat(batch_num,1,1,1).cuda()

        #bilateral voting
        vote_out = self.Bilater_voting(c_map)

        #global map
        glo_map, _ = torch.max(vote_out,dim=2)
        vote_out = vote_out.squeeze(1)
        #decouple loss
        de_loss = edge_loss(glo_map,vote_out,edge,target)

        conmap_l = self.cross_entropy_loss(c_map,con_target)
        bimap_l = self.cross_entropy_loss(vote_out,con_target)
        loss =  0.8*conmap_l+de_loss+0.2*bimap_l

        return loss
    
    def shift_diag(self,img,shift):
        ## shift = [1,1] moving right and down
        # print(img.shape,self.hori_translation.shape)
        batch,class_num, row, column = img.size()

        if shift[0]: ###horizontal
            img = torch.bmm(img.view(-1,row,column),self.hori_translation.view(-1,column,column)) if shift[0]==1 else torch.bmm(img.view(-1,row,column),self.hori_translation.transpose(3,2).view(-1,column,column))
        if shift[1]: ###vertical
            img = torch.bmm(self.verti_translation.transpose(3,2).view(-1,row,row),img.view(-1,row,column)) if shift[1]==1 else torch.bmm(self.verti_translation.view(-1,row,row),img.view(-1,row,column))
        return img.view(batch,class_num, row, column)


    def Bilater_voting(self,c_map):
        c_map = c_map.view(c_map.shape[0],-1,8,c_map.shape[2],c_map.shape[3])
        batch,class_num,channel, row, column = c_map.size()


        shifted_c_map = torch.zeros(c_map.size()).cuda()
        for i in range(8):
            shifted_c_map[:,:,i] = self.shift_diag(c_map[:,:,7-i].clone(),Directable[TL_table[i]])
        vote_out = c_map*shifted_c_map

#         pred_mask,_ = torch.max(vote_out,dim=2)
        # print(pred_mask)
        return vote_out#, bimap
