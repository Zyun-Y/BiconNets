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

def Bilater_voting(c_map,hori_translation,verti_translation):

    
    hori_translation = hori_translation.cuda()
    # print(hori_translation)
    verti_translation = verti_translation.cuda()
    # print(hori_translation.shape)
    batch,channel, row, column = c_map.size()
    vote_out = torch.zeros([batch,channel, row, column]).cuda()

    eps = 0
    # print(c_map[1,4].shape)
    right = torch.bmm(c_map[:,4],hori_translation)
    left = torch.bmm(c_map[:,3],hori_translation.transpose(2,1))
    left_bottom = torch.bmm(verti_translation.transpose(2,1), c_map[:,5])
    left_bottom = torch.bmm(left_bottom,hori_translation.transpose(2,1))
    right_above = torch.bmm(verti_translation, c_map[:,2])
    right_above= torch.bmm(right_above,hori_translation)
    left_above = torch.bmm(verti_translation, c_map[:,0])
    left_above = torch.bmm(left_above,hori_translation.transpose(2,1))
    bottom = torch.bmm(verti_translation.transpose(2,1), c_map[:,6])
    up = torch.bmm(verti_translation, c_map[:,1])
    right_bottom = torch.bmm(verti_translation.transpose(2,1), c_map[:,7])
    right_bottom = torch.bmm(right_bottom,hori_translation)
    # print(right[0][0][100])
    # print(c_map[:,3][0][0][100])
    a1 = (c_map[:,3]) * (right)

    # print(a1[0][0][100])
    a2 = (c_map[:,4]) * (left)
    a3 = (c_map[:,1]) * (bottom)
    a4 = (c_map[:,6]) * (up+eps)
    a5 = (c_map[:,2]) * (left_bottom)
    a6 = (c_map[:,5]) * (right_above)
    a7 =(c_map[:,0]) * (right_bottom)
    a8 =(c_map[:,7]) * (left_above)
    vote_out[:,0] = a7
    vote_out[:,1] = a3
    vote_out[:,2] = a5
    vote_out[:,3] = a1
    vote_out[:,4] = a2
    vote_out[:,5] = a6
    vote_out[:,6] = a4
    vote_out[:,7] = a8
    # pred_mask = torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(a1,a2),a3),a4),a5),a6),a7),a8)
    pred_mask = torch.mean(vote_out,dim=1)
    # print(pred_mask[1])
    return pred_mask,vote_out



def edge_loss(glo_map,vote_out,edge,target):
    pred_mask_min, _ = torch.min(vote_out, dim=1)
    pred_mask_min = 1-pred_mask_min
    pred_mask_min = pred_mask_min * edge
    decouple_map = glo_map*(1-edge)+pred_mask_min

    de_loss = F.binary_cross_entropy(decouple_map.unsqueeze(1),target)

    return de_loss

class bicon_loss(nn.Module):
    def __init__(self):
        super(bicon_loss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss()
        

    def forward(self, atts,dets, target, con_target):
        con_target = con_target.type(torch.FloatTensor).cuda()


        hori_translation = torch.zeros([atts.shape[0],atts.shape[3],atts.shape[3]])
        for i in range(atts.shape[3]-1):
            hori_translation[:,i,i+1] = torch.tensor(1.0)
        verti_translation = torch.zeros([atts.shape[0],atts.shape[2],atts.shape[2]])
        for j in range(atts.shape[2]-1):
            verti_translation[:,j,j+1] = torch.tensor(1.0)
        hori_translation = hori_translation.float()
        verti_translation = verti_translation.float()

        sum_conn = torch.sum(con_target,dim=1)
        edge_conn = torch.where(sum_conn<8,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        edge0 = torch.where(sum_conn>0,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        edge_conn = edge_conn*edge0

        target = target.type(torch.FloatTensor).cuda()
        atts = F.sigmoid(atts)
        dets = F.sigmoid(dets)
        

        glo_map1,vote_out1 = Bilater_voting(atts,hori_translation,verti_translation)
        glo_map2,vote_out2 = Bilater_voting(dets,hori_translation,verti_translation)


        decouple_loss2 = edge_loss(glo_map2,vote_out2,edge_conn,target) 

        bce_loss1 = self.cross_entropy_loss(glo_map1.unsqueeze(1), target)

        bicon_loss1 = self.cross_entropy_loss(vote_out1,con_target)
        conn_loss1 = self.cross_entropy_loss(atts,con_target)

        bicon_loss2 = self.cross_entropy_loss(vote_out2,con_target)
        conn_loss2 = self.cross_entropy_loss(dets,con_target)

        loss =  0.2*bicon_loss1+0.8*conn_loss1+bce_loss1  + 0.2*bicon_loss2+0.8*conn_loss2+decouple_loss2



        return loss

        
