#!/usr/bin/python3
#coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import glob
from PIL import Image
from utils_bicon import *
import torch.nn.functional as F
########################### Data Augmentation ###########################

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std 
    
    def __call__(self, image, mask):
        image = (image - self.mean)/self.std
        mask /= 255
        return image, mask

class RandomCrop(object):
    def __call__(self, image, mask, conn,size):
        H,W,_   = image.shape
        offsetH = H-size[0]
        offsetW = W-size[0]

        randh   = np.random.randint(offsetH)
        randw   = np.random.randint(offsetW)

        h_end = randh+size[0]
        w_end = randw+size[1]

        return image[randh:h_end,randw:w_end, :], mask[randh:h_end,randw:w_end], conn[:,randh:h_end,randw:w_end]

class RandomCrop_pool(object):
    def __call__(self, image, mask, conn):
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        return image[p0:p1,p2:p3, :], mask[p0:p1,p2:p3], conn[:,p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask,conn):
        if np.random.randint(2)==0:
            return image[:,::-1,:], mask[:, ::-1], conn[:,:, ::-1]
        else:
            return image, mask,conn

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask,mode):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mode == 'train':
            conn = sal2conn(mask)
            # print(conn.shape)
            # print(mask.shape)
            return image, mask,conn
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask,conn):
        image = torch.Tensor(image.copy())
        image = image.permute(2, 0, 1)
        mask  = torch.Tensor(mask.copy())
        conn  = torch.Tensor(conn.copy())
        return image, mask, conn

class ToTensor_test(object):
    def __call__(self, image, mask):
        image = torch.Tensor(image.copy())
        image = image.permute(2, 0, 1)
        mask  = torch.Tensor(mask.copy())

        return image, mask


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean   = np.array([[[124.55, 118.90, 102.94]]])
        self.std    = np.array([[[ 56.77,  55.97,  57.50]]])


        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s'%(k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg, mode):
        self.cfg        = cfg
        
        self.randomcrop = RandomCrop()
        self.randomcrop_pool = RandomCrop_pool()
        self.randomflip = RandomFlip()
        self.resize     = Resize(320, 320)
        self.totensor   = ToTensor()
        self.totensor_test=ToTensor_test()
        self.img_ls = []
        self.mask_ls=[]
        self.conn_ls = []

        
        
        

        # image_ls_train = image_ls[0:exp_id*500]+image_ls[exp_id*500+500:]
        # print(len(image_ls_train))
        if self.cfg.mode=='train':
            if mode == 'norm':
                mean   = np.array([[[124.55, 118.90, 102.94]]])
                std    = np.array([[[ 56.77,  55.97,  57.50]]])
            else:
                mean   = np.array([[[124.55, 118.90, 102.94]]])
                std    = np.array([[[ 56.77,  55.97,  57.50]]])
            self.normalize  = Normalize(mean=mean, std=std)
            image_ls = glob.glob(cfg.datapath+'/imgs/*.jpg')
            for im in image_ls:
                # print(im)
                self.img_ls.append(im)
                mask = im.split('imgs')[0]+'gt_128'+im.split('imgs')[1].split('.jpg')[0]+'.png'
                self.mask_ls.append(mask)
                conn = im.split('imgs')[0]+'connect_128'+im.split('imgs')[1].split('.jpg')[0]+'.npy'
                self.conn_ls.append(conn)

        if self.cfg.mode=='DUTS':
            if mode == 'norm':
                mean   = np.array([[[124.55, 118.90, 102.94]]])
                std    = np.array([[[ 56.77,  55.97,  57.50]]])
            else:
                mean   = np.array([[[124.55, 118.90, 102.94]]])
                std    = np.array([[[ 56.77,  55.97,  57.50]]])
            self.normalize  = Normalize(mean=mean, std=std)
            image_ls = glob.glob(cfg.datapath+'/imgs/*.jpg')
            for im in image_ls:
                # print(im)
                self.img_ls.append(im)
                mask = im.split('imgs')[0]+'gt'+im.split('imgs')[1].split('.jpg')[0]+'.png'
                self.mask_ls.append(mask)

        if self.cfg.mode=='DUT-O':
            if mode == 'norm':
                mean   = np.array([[[124.55, 118.90, 102.94]]])
                std    = np.array([[[ 56.77,  55.97,  57.50]]])
            else:
                mean      = np.array([[[120.61, 121.86, 114.92]]])
                std       = np.array([[[ 58.10,  57.16,  61.09]]])
            self.normalize  = Normalize(mean=mean, std=std)
            image_ls = glob.glob(cfg.datapath+'/imgs/*.jpg')
            for im in image_ls:
                # print(im)
                self.img_ls.append(im)
                mask = im.split('imgs')[0]+'gt'+im.split('imgs')[1].split('.jpg')[0]+'.png'
                self.mask_ls.append(mask)

        if self.cfg.mode=='HKU':
            if mode == 'norm':
                mean   = np.array([[[124.55, 118.90, 102.94]]])
                std    = np.array([[[ 56.77,  55.97,  57.50]]])
            else:
                mean      = np.array([[[123.58, 121.69, 104.22]]])
                std       = np.array([[[ 55.40,  53.55,  55.19]]])
            self.normalize  = Normalize(mean=mean, std=std)
            image_ls = glob.glob(cfg.datapath+'/imgs/*.png')
            for im in image_ls:
                # print(im)
                self.img_ls.append(im)
                mask = im.split('imgs')[0]+'gt'+im.split('imgs')[1].split('.png')[0]+'.png'
                self.mask_ls.append(mask)

        if self.cfg.mode=='PASCAL':
            if mode == 'norm':
                mean   = np.array([[[124.55, 118.90, 102.94]]])
                std    = np.array([[[ 56.77,  55.97,  57.50]]])
            else:
                mean      = np.array([[[117.02, 112.75, 102.48]]])
                std       = np.array([[[ 59.81,  58.96,  60.44]]])
            self.normalize  = Normalize(mean=mean, std=std)
            image_ls = glob.glob(cfg.datapath+'/imgs/*.jpg')
            for im in image_ls:
                # print(im)
                self.img_ls.append(im)
                mask = im.split('imgs')[0]+'gt'+im.split('imgs')[1].split('.jpg')[0]+'.png'
                self.mask_ls.append(mask)

        if self.cfg.mode=='ECSSD':
            if mode == 'norm':
                mean   = np.array([[[124.55, 118.90, 102.94]]])
                std    = np.array([[[ 56.77,  55.97,  57.50]]])
            else:
                mean      = np.array([[[117.15, 112.48, 92.86]]])
                std       = np.array([[[ 56.36,  53.82, 54.23]]])
            self.normalize  = Normalize(mean=mean, std=std)
            image_ls = glob.glob(cfg.datapath+'/imgs/*.jpg')
            for im in image_ls:
                # print(im)
                self.img_ls.append(im)
                mask = im.split('imgs')[0]+'gt'+im.split('imgs')[1].split('.jpg')[0]+'.png'
                self.mask_ls.append(mask)

        if self.cfg.mode=='sod':
            if mode == 'norm':
                mean   = np.array([[[124.55, 118.90, 102.94]]])
                std    = np.array([[[ 56.77,  55.97,  57.50]]])
            else:
                mean      = np.array([[[109.91, 112.13,  93.90]]])
                std       = np.array([[[ 53.29,  50.45,  48.06]]])
            self.normalize  = Normalize(mean=mean, std=std)
            image_ls = glob.glob(cfg.datapath+'/imgs/*.jpg')
            # print(cfg.datapath)
            for im in image_ls:
                # print(im)
                self.img_ls.append(im)
                mask = im.split('.jpg')[0]+'.png'
                # print(mask)
                self.mask_ls.append(mask)




    def __getitem__(self, idx):

        image = cv2.imread(self.img_ls[idx])[:,:,::-1].astype(np.float32)
        mask  = cv2.imread(self.mask_ls[idx], 0).astype(np.float32)
        name = self.mask_ls[idx].split('/gt/')[1]

        shape = mask.shape


        if self.cfg.mode=='train':
            conn = np.load(self.conn_ls[idx]).astype(np.float32)
            image, mask = self.normalize(image, mask)
            image, mask,conn = self.randomcrop_pool(image, mask,conn)
            image, mask,conn = self.randomflip(image, mask,conn)
            image, mask,conn =self.totensor(image, mask,conn)
            return image, mask.unsqueeze(0), conn
        else:
            image, im_size = load_image_test( self.img_ls[idx])
            image = torch.Tensor(image)
            mask = load_sal_label(self.mask_ls[idx])
            mask = torch.Tensor(mask)


            return image, mask,name


    def __len__(self):
        return len(self.img_ls)

def load_image(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_

def load_image_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_, im_size

def load_sal_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label

    

########################### Testing Script ###########################
if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.ion()

    cfg  = Config(mode='train', datapath='/home/ziyun/Desktop/Project/data/DUTS/DUTS-TR')
    data = Data(cfg)
    for i in range(1000):
        image, mask = data[i]
        image       = image*cfg.std + cfg.mean
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()
