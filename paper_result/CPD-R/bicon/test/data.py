import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch

class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, trainsize):
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        # print(gt.shape, gt.max())
        gt = gt.ge(0.5).float()
        conn = connectivity_matrix(gt.squeeze())
        return image, gt, conn

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, trainsize,test_root,test_gt_root, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    tst_dataset = test_dataset(test_root, test_gt_root, trainsize)
    test_loader = data.DataLoader(dataset=tst_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader, test_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        # self.index = 0

    def __getitem__(self,index):
        image = self.rgb_loader(self.images[index])
        image = self.transform(image)
        gt = self.gt_transform(self.binary_loader(self.gts[index]))
        name = self.images[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


def connectivity_matrix(mask):
    [rows, cols] = mask.shape

    conn = torch.zeros([8,rows, cols])
    up = torch.zeros([rows, cols])#move the orignal mask to up
    down = torch.zeros([rows, cols])
    left = torch.zeros([rows, cols])
    right = torch.zeros([rows, cols])
    up_left = torch.zeros([rows, cols])
    up_right = torch.zeros([rows, cols])
    down_left = torch.zeros([rows, cols])
    down_right = torch.zeros([rows, cols])


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
    return conn