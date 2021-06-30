import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import glob
import numpy as np
import random
from utils_bicon import *
root = '/data/DUTS/DUTS-TR'
ECSSD = '/data/ECSSD'
pascal = '/data/PASCALS'
omron = '/data/DUTS/DUT-OMRON'
hku = '/data/HKU-IS'
duts = '/data/DUTS/DUTS-TE'
sod = '/data/SOD'
data_dict = {'train':root, 'ECSSD':ECSSD, 'PASCAL': pascal, 'HKU':hku, 'DUTS':duts,'DUT-O':omron, 'sod':sod}

class ImageDataTrain(data.Dataset):
    def __init__(self, data_root):
        image_ls = glob.glob(data_root+'/edge/*.png')
        self.img_ls = []
        self.mask_ls = []
        # self.conn_ls = []
        self.edge_ls = []
        for edge in image_ls:
            # print(im)
            
            im = edge.split('/edge/')[0]+'/imgs/'+edge.split('/edge/')[1].split('.png')[0]+'.jpg'
            mask = edge.split('/edge/')[0]+'/gt/'+edge.split('/edge/')[1].split('.png')[0]+'.png'
            self.edge_ls.append(edge)
            self.img_ls.append(im)
            self.mask_ls.append(mask)


    def __getitem__(self, item):
        # sal data loading
        im_name = self.img_ls[item]
        gt_name = self.mask_ls[item]
        edge_name = self.edge_ls[item]

        # conn_name = self.conn_ls[item]
        sal_image = load_image(im_name)
        sal_label = load_sal_label(gt_name)
        edge = load_sal_edge(edge_name)


        # conn = np.load(conn_name)
        # conn = np.array(conn, dtype=np.float32)

        sal_image, sal_label, edge = cv_random_flip(sal_image, sal_label, edge)
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        edge = torch.Tensor(edge)
        # print(sal_label.shape)
        conn = sal2conn(sal_label)
        conn = torch.Tensor(conn)

        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'sal_conn':conn, 'sal_edge':edge}
        return sample

    def __len__(self):
        return len(self.img_ls)

class ImageDataTest(data.Dataset):
    def __init__(self, data_root,root):
        if root == 'HKU':
            post = '.png'
        else:
            post = '.jpg'
        image_ls = glob.glob(data_root+'/imgs/*'+post)
        self.img_ls = []
        self.mask_ls = []
        self.img_name = []
        for im in image_ls:
            # print(im)
            self.img_ls.append(im)
            mask = im.split('imgs')[0]+'gt'+im.split('imgs')[1].split(post)[0]+'.png'
            self.mask_ls.append(mask)
            self.name = im.split('imgs/')[1].split(post)[0]
            self.img_name.append(self.name)
            # print(self.img_name)
    def __getitem__(self, item):
        image, im_size = load_image_test( self.img_ls[item])
        label = load_sal_label(self.mask_ls[item])
        image = torch.Tensor(image)
        label = torch.Tensor(label)
        return {'image': image, 'label':label, 'name': self.img_name[item], 'size': im_size}

    def __len__(self):
        return len(self.img_ls)


def get_loader(root,batch_size, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(data_dict[root])
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=pin)
    else:
        print(data_dict[root])
        dataset = ImageDataTest(data_dict[root],root)
        data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin)
    return data_loader, dataset

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
    """
    pixels > 0.5 -> 1
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    """
    if not os.path.exists(path):
        print('File Not Exists')
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    # label = cv2.resize(label, im_sz, interpolation=cv2.INTER_NEAREST)
    label = label / 255.
    label[np.where(label > 0.5)] = 1.
    label = label[np.newaxis, ...]
    return label

def load_sal_edge(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label

def cv_random_flip(img, label, edge):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
        # conn = conn[:,:,::-1].copy()
        edge = edge[:,:,::-1].copy()
    return img, label, edge
