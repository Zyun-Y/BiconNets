from PIL import Image
import numpy as np
import time

import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F

def ConMap2Mask_prob(c_map,hori_translation,verti_translation):
    # print(c_map.shape)
    c_map = F.sigmoid(c_map)
    hori_translation = hori_translation.cuda()
    # print(hori_translation)
    verti_translation = verti_translation.cuda()
    # print(hori_translation.shape)
    batch,channel, row, column = c_map.size()
    vote_out = torch.zeros([batch,channel, row, column]).cuda()

    eps = 0
    # print(c_map[:,4].shape,hori_translation.shape)
    # print(c_map.shape)
    right = torch.bmm(c_map[:,4],hori_translation)
    left = torch.bmm(c_map[:,3],hori_translation.transpose(2,1))
    # print(verti_translation.shape,c_map[:,5].shape)
    # print(verti_translation.transpose(2,1).shape,c_map[:,5].shape)
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

    a1 = (c_map[:,3]) * (right)

    # print(a1[0][0][100])
    a2 = (c_map[:,4]) * (left)
    a3 = (c_map[:,1]) * (bottom )
    a4 = (c_map[:,6]) * (up)
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
    # vote_out = vote_out.cuda()
    # pred_mask = torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(torch.max(a1,a2),a3),a4),a5),a6),a7),a8)
    # pred_mask_sum = a1+a2+a3+a4+a5+a6+a7+a8
    pred_mask = torch.mean(vote_out, dim=1)
    # print(pred_mask[1])
    # pred_mask = pred_mask.unsqueeze(1)
    return pred_mask,vote_out

class PRF(nn.Module):
    def __init__(self, device, Y, steps=255, end=1):
        super(PRF, self).__init__()
        self.thresholds = torch.linspace(0, end, steps=steps).cuda(device)
        self.Y = Y

    def forward(self, _Y):
        TPs =  [torch.sum(torch.sum((_Y >= threshold).int() & (self.Y).int(), -1).int(), -1).float() for threshold in self.thresholds]
        T1s = [torch.sum(torch.sum(_Y >= threshold, -1), -1).float() for threshold in self.thresholds]
        T2 = Tensor.float(torch.sum(torch.sum(self.Y, -1), -1))
        Ps = [(TP / (T1 + 1e-9)).mean() for TP, T1 in zip(TPs, T1s)]
        Rs = [(TP / (T2 + 1e-9)).mean() for TP in TPs]
        Fs = [(1.3 * P * R / (R + 0.3 * P + 1e-9)) for P, R in zip(Ps, Rs)]

        return {'P':Ps, 'R':Rs, 'F':Fs}
        
def getOutPuts(model, DX, args, supervised=False):
    num_img, channel, height, width = DX.shape
    # print(DX.shape)
    if supervised:
        OutPuts = {'final':np.empty((len(DX), height, width), dtype=np.float32), 'contour':np.empty((len(DX), 5, height, width), dtype=np.float32), 'preds':np.empty((len(DX), 5, height, width), dtype=np.float32), 'time':0.}
    else:
        OutPuts = {'final':np.empty((len(DX), height, width), dtype=np.float32), 'time':0.} 
    t1 = time.time()

    for idx in range(0, len(DX), args.batch):
        # print(idx)
        ind = min(len(DX), idx + args.batch)
        X = torch.tensor(DX[idx:ind]).cuda(args.ids[0]).float()
        # print(X.shape)
        Outs = model(X)

        hori_translation = torch.zeros([Outs['final'].shape[0],Outs['final'].shape[3],Outs['final'].shape[3]])
        for i in range(Outs['final'].shape[3]-1):
            hori_translation[:,i,i+1] = torch.tensor(1.0)
        verti_translation = torch.zeros([Outs['final'].shape[0],Outs['final'].shape[2],Outs['final'].shape[2]])
        # print(verti_translation.shape)
        for j in range(Outs['final'].shape[2]-1):
            verti_translation[:,j,j+1] = torch.tensor(1.0)
        hori_translation = hori_translation.float().cuda()
        verti_translation = verti_translation.float().cuda()

        Outs['final'],_ =ConMap2Mask_prob(Outs['final'],hori_translation,verti_translation)

        # print(Outs['final'].shape)
        OutPuts['final'][idx:ind] = Outs['final'].cpu().data.numpy()
        
        if supervised:
            for supervision in ['preds', 'contour']:
                for i, pred in enumerate(Outs[supervision]):
                    pre = F.interpolate(pred.unsqueeze(0), (height, width), mode='bilinear')[0]
                    pre = torch.sigmoid(pre).cpu().data.numpy()
                    OutPuts[supervision][idx:ind, i] = pre
        
        X, Outs, pre = 0, 0, 0

    OutPuts['time'] = (time.time() - t1)

        
    return OutPuts


def mae(preds, labels, th=0.5):
    return np.mean(np.abs(preds - labels))

def fscore(preds, labels, th=0.5):
    tmp = preds >= th
    TP = np.sum(tmp & labels)
    T1 = np.sum(tmp)
    T2 = np.sum(labels)
    F = 1.3 * TP / (T1 + 0.3 * T2 + 1e-9)

    return F

def maxF(preds, labels, device):
    preds = torch.tensor(preds).cuda(device)
    labels = torch.tensor(labels, dtype=torch.uint8).cuda(device)
    
    prf = PRF(device, labels).cuda(device)
    Fs = prf(preds)['F']
    Fs = [F.cpu().data.numpy() for F in Fs]

    prf.to(torch.device('cpu'))
    torch.cuda.empty_cache()
    return max(Fs)

def Normalize(atten):

    a_min, a_max = atten.min(), atten.max()
    atten = (atten - a_min) * 1. / (a_max - a_min) * 255.

    return np.uint8(atten)
