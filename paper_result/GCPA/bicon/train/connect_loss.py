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



def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    # pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def edge_loss(glo_map,vote_out,edge,target):
    # print(glo_map.shape,vote_out.shape,edge.shape,target.shape)
    pred_mask_min, _ = torch.min(vote_out, dim=1)
    pred_mask_min = 1-pred_mask_min
    pred_mask_min = pred_mask_min * edge
    decouple_map = glo_map*(1-edge)+pred_mask_min

    minloss = F.binary_cross_entropy(decouple_map.unsqueeze(1),target)

    return minloss

class bicon_loss(nn.Module):
    def __init__(self):
        super(bicon_loss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss()
        

    def forward(self, out2, out3, out4, out5, target, con_target):
        con_target = con_target.type(torch.FloatTensor).cuda()


        hori_translation = torch.zeros([out2.shape[0],out2.shape[3],out2.shape[3]])
        for i in range(out2.shape[3]-1):
            hori_translation[:,i,i+1] = torch.tensor(1.0)
        verti_translation = torch.zeros([out2.shape[0],out2.shape[2],out2.shape[2]])
        for j in range(out2.shape[2]-1):
            verti_translation[:,j,j+1] = torch.tensor(1.0)
        hori_translation = hori_translation.float()
        verti_translation = verti_translation.float()

        target = target.type(torch.FloatTensor).cuda()
        out2 = F.sigmoid(out2)
        out3 = F.sigmoid(out3)
        out4 = F.sigmoid(out4)
        out5 = F.sigmoid(out5)

        sum_conn = torch.sum(con_target,dim=1)
        edge = torch.where(sum_conn<8,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        edge0 = torch.where(sum_conn>0,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        edge = edge*edge0


        glo_map1,vote_out1 = Bilater_voting(out2,hori_translation,verti_translation)
        glo_map2,vote_out2 = Bilater_voting(out3,hori_translation,verti_translation)
        glo_map3,vote_out3 = Bilater_voting(out4,hori_translation,verti_translation)
        glo_map4,vote_out4 = Bilater_voting(out5,hori_translation,verti_translation)


        de_loss1 = edge_loss(glo_map1,vote_out1,edge,target)
        de_loss2 = edge_loss(glo_map2,vote_out2,edge,target)
        de_loss3 = edge_loss(glo_map3,vote_out3,edge,target)
        de_loss4 = edge_loss(glo_map4,vote_out4,edge,target)

        de_loss   =de_loss1+0.8*(de_loss2)+0.6*(de_loss3)+0.4*(de_loss4)


        conmmap_l1 = self.cross_entropy_loss(out2,con_target)
        bimap_l1 = self.cross_entropy_loss(vote_out1,con_target)

        conmmap_l2 = self.cross_entropy_loss(out3,con_target)
        bimap_l2 = self.cross_entropy_loss(vote_out2,con_target)

        conmmap_l3 = self.cross_entropy_loss(out4,con_target)
        bimap_l3 = self.cross_entropy_loss(vote_out3,con_target)

        conmmap_l4 = self.cross_entropy_loss(out5,con_target)
        bimap_l4 = self.cross_entropy_loss(vote_out4,con_target)

        loss =  0.8*conmmap_l1+0.2*bimap_l1+0.64*conmmap_l2+0.16*bimap_l2+0.48*conmmap_l3+0.12*bimap_l3+0.32*conmmap_l4+0.08*bimap_l4+de_loss

        return loss

        
