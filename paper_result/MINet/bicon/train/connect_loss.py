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




class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        # pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)

def edge_loss(glo_map,vote_out,edge,target):
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
        self.cel = CEL()
        

    def forward(self, out, target, con_target):
        con_target = con_target.type(torch.FloatTensor).cuda()

        hori_translation = torch.zeros([out.shape[0],out.shape[3],out.shape[3]])
        for i in range(out.shape[3]-1):
            hori_translation[:,i,i+1] = torch.tensor(1.0)
        verti_translation = torch.zeros([out.shape[0],out.shape[2],out.shape[2]])
        for j in range(out.shape[2]-1):
            verti_translation[:,j,j+1] = torch.tensor(1.0)
        hori_translation = hori_translation.float()
        verti_translation = verti_translation.float()
        target = target.type(torch.FloatTensor).cuda()

        out = F.sigmoid(out)

        sum_conn = torch.sum(con_target,dim=1)
        edge = torch.where(sum_conn<8,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        edge0 = torch.where(sum_conn>0,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        edge = edge*edge0

        # print(out.    shape,hori_translation.shape,verti_translation.shape)
        glo_map,vote_out = Bilater_voting(out,hori_translation,verti_translation)




        decouple_loss = edge_loss(glo_map,vote_out,edge,target)

        cel_loss = self.cel(glo_map.unsqueeze(1),target)

            
        loss   =decouple_loss + cel_loss

        bimap_loss = self.cross_entropy_loss(vote_out,con_target)
        conmap_loss = self.cross_entropy_loss(out,con_target)


        loss =  0.8*conmap_loss+0.2*bimap_loss+loss


        return loss
