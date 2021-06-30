import os
import cv2
import torch
import glob
import numpy as np
import torch.nn.functional as F

def sal2conn(mask):
    ## converte the saliency mask into a connectivity mask
    ## mask shape: H*W, output connectivity shape: 8*H*W
    [rows, cols] = mask.shape
    conn = np.zeros([8,rows, cols])
    up = np.zeros([rows, cols])#move the orignal mask to up
    down = np.zeros([rows, cols])
    left = np.zeros([rows, cols])
    right = np.zeros([rows, cols])
    up_left = np.zeros([rows, cols])
    up_right = np.zeros([rows, cols])
    down_left = np.zeros([rows, cols])
    down_right = np.zeros([rows, cols])


    up[:rows-1, :] = mask[1:rows,:]
    down[1:rows,:] = mask[0:rows-1,:]
    left[:,:cols-1] = mask[:,1:cols]
    right[:,1:cols] = mask[:,:cols-1]
    up_left[0:rows-1,0:cols-1] = mask[1:rows,1:cols]
    up_right[0:rows-1,1:cols] = mask[1:rows,0:cols-1]
    down_left[1:rows,0:cols-1] = mask[0:rows-1,1:cols]
    down_right[1:rows,1:cols] = mask[0:rows-1,0:cols-1]

    conn[0] = mask*down_right
    conn[1] = mask*down
    conn[2] = mask*down_left
    conn[3] = mask*right
    conn[4] = mask*left
    conn[5] = mask*up_right
    conn[6] = mask*up
    conn[7] = mask*up_left
    conn = conn.astype(np.float32)
    return conn



def bv_test(output_test):
    '''
    generate the continous global map from output connectivity map as final saliency output 
    via bilateral voting
    '''

    #construct the translation matrix
    hori_translation = torch.zeros([output_test.shape[0],output_test.shape[3],output_test.shape[3]])
    for i in range(output_test.shape[3]-1):
        hori_translation[:,i,i+1] = torch.tensor(1.0)
    verti_translation = torch.zeros([output_test.shape[0],output_test.shape[2],output_test.shape[2]])
    # print(verti_translation.shape)
    for j in range(output_test.shape[2]-1):
        verti_translation[:,j,j+1] = torch.tensor(1.0)

    hori_translation = hori_translation.float().cuda()
    verti_translation = verti_translation.float().cuda()

    pred =ConMap2Mask_prob(output_test,hori_translation,verti_translation)
    return pred


def ConMap2Mask_prob(c_map,hori_translation,verti_translation):
    '''
    continuous bilateral voting
    '''
    c_map = F.sigmoid(c_map)
    hori_translation = hori_translation.cuda()
    # print(hori_translation)
    verti_translation = verti_translation.cuda()
    # print(hori_translation.shape)
    batch,channel, row, column = c_map.size()
    vote_out = torch.zeros([batch,channel, row, column]).cuda()

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

    a1 = (c_map[:,3]) * (right)
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
    pred_mask = torch.mean(vote_out,dim=1)
    pred_mask = pred_mask.unsqueeze(1)
    return pred_mask