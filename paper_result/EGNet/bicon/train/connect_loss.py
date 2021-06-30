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


def bce2d_new(input, target, reduction=None):

    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

def edge_loss(glo_map,vote_out,edge,target):
    # print(glo_map.shape,vote_out.shape,edge.shape,target.shape)
    pred_mask_min, _ = torch.min(vote_out, dim=1)
    pred_mask_min = 1-pred_mask_min
    pred_mask_min = pred_mask_min * edge
    decouple_map = glo_map*(1-edge)+pred_mask_min

    minloss = F.binary_cross_entropy(decouple_map.unsqueeze(1),target,reduction='sum')

    return minloss

class bicon_loss(nn.Module):
    def __init__(self):
        super(bicon_loss, self).__init__()
        self.cross_entropy_loss = nn.BCELoss(reduction='sum')
        

    def forward(self,  up_edge, up_sal, up_sal_f, target,edge, con_target,weight,voted, unbalance):
        con_target = con_target.type(torch.FloatTensor).cuda()

        hori_translation = torch.zeros([target.shape[0],target.shape[3],target.shape[3]])
        for i in range(target.shape[3]-1):
            hori_translation[:,i,i+1] = torch.tensor(1.0)
        verti_translation = torch.zeros([target.shape[0],target.shape[2],target.shape[2]])
        for j in range(target.shape[2]-1):
            verti_translation[:,j,j+1] = torch.tensor(1.0)
        hori_translation = hori_translation.float()
        verti_translation = verti_translation.float()


        target = target.type(torch.FloatTensor).cuda()

        sum_conn = torch.sum(con_target,dim=1)
        edge_gt = torch.where(sum_conn<8,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        edge0_gt = torch.where(sum_conn>0,torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        edge = edge_gt*edge0_gt

        r_edge_loss, r_sal_loss, r_sum_loss= 0,0,0

        # edge part
        edge_loss_egnet = []
        for ix in up_edge:

            edge_loss_egnet.append(bce2d_new(ix, edge.unsqueeze(1), reduction='sum'))
        edge_loss_egnet = sum(edge_loss_egnet)# / real_batch
        r_edge_loss += edge_loss_egnet.data


        # sal part
        sal_loss1= []
        sal_loss2 = []
        for ix in up_sal:
            ix = F.sigmoid(ix)
            glo_map,vote_out = Bilater_voting(ix,hori_translation,verti_translation)

            decouple_l = edge_loss(glo_map,vote_out,edge,target)

            bimap_l1 = self.cross_entropy_loss(vote_out,con_target)
            conmap_l1 = self.cross_entropy_loss(ix,con_target)



            sal_loss1.append(decouple_l+0.2*bimap_l1+0.8*conmap_l1)

        for ix_1 in up_sal_f:
            # print(ix_1.shape)
            ix_1 = F.sigmoid(ix_1)
            glo_map_1,vote_out_1 = Bilater_voting(ix_1,hori_translation,verti_translation)
            decouple_l2 = edge_loss(glo_map_1,vote_out_1,edge,target)

            bimap_l2 = self.cross_entropy_loss(vote_out_1,con_target)
            conmap_l2 = self.cross_entropy_loss(ix_1,con_target)


            sal_loss2.append(decouple_l2+0.2*bimap_l2+0.8*conmap_l2)



        sal_loss = (sum(sal_loss1) + sum(sal_loss2))# / real_batch
      
        r_sal_loss += sal_loss.data
        loss = sal_loss + edge_loss_egnet
        r_sum_loss += loss.data

        return loss,r_edge_loss, r_sal_loss, r_sum_loss

