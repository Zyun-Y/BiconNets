import torch
import torch.nn.functional as F
from torch.autograd import Variable
import imageio
import numpy as np
import pdb, os, argparse
from datetime import datetime
from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from data import get_loader
from utils import clip_gradient, adjust_lr
from utils_bicon import *
torch.cuda.set_device(1)
save = 'save'
if not os.path.exists(save):
    os.makedirs(save)
root = '/data/DUTS/DUTS-TR'
ECSSD = '/data/ECSSD'
pascal = '/data/PASCALS'
omron = '/data/DUTS/DUT-OMRON'
hku = '/data/HKU-IS'
duts = '/data/DUTS/DUTS-TE'

data_dict = {'train':root, 'ECSSD':ECSSD, 'PASCAL': pascal, 'HKU':hku, 'DUTS':duts,'DUT-O':omron}

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=4e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=True, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models
if opt.is_ResNet:
    model = CPD_ResNet()
else:
    model = CPD_VGG()

model.cuda()


def test(test_loader,model,epoch,exp_id):
        model.eval()
        with torch.no_grad(): 
            for j_batch, test_data in enumerate(test_loader):
                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])
                name = test_data[2]
                X_test= X_test.cuda()
                y_test = y_test.cuda()
                # print(X_test.shape,y_test.shape)
                if X_test.shape[1]==1:
                    X_test = torch.cat([X_test,X_test,X_test],1)
                # print(X_test.shape,y_test.shape)
                _, output_test = model(X_test)

                pred = bv_test(output_test)
                
                pred = F.upsample(pred, size=(y_test.shape[2],y_test.shape[3]), mode='bilinear', align_corners=False)
                
                for im in range(y_test.shape[0]):

                    save_fig(pred[im][0],y_test[im][0],exp_id,name[0])


def save_fig(pred,y_test,exp_id,name):
    path = 'output/'+str(exp_id)
    path1 = 'output/'+str(exp_id)+'/gt'
    path2 = 'output/'+str(exp_id)+'/pred'
    if not os.path.exists(path1):
        os.makedirs(path)
        os.makedirs(path1)
        os.makedirs(path2)

    imageio.imwrite('output/'+str(exp_id)+'/pred/'+name+'.png', pred.cpu().numpy())
    imageio.imwrite('output/'+str(exp_id)+'/gt/'+name+'.png', y_test.cpu().numpy())

print("Let's go!")
device = torch.device('cpu')
model.load_state_dict(torch.load('/CPD_model.pth',map_location=device))
model.cuda()
for exp_id in range(6):
    if exp_id ==3:
        dataset = 'PASCAL'
    if exp_id ==1: 
        dataset = 'HKU'
    if exp_id ==2:
        dataset = 'DUT-O'
    if exp_id ==4:
        dataset = 'DUTS'
    if exp_id ==0:
        dataset = 'ECSSD'

    test_path = data_dict[dataset]
    image_root = '/data/DUTS/DUTS-TR/imgs/'
    gt_root = '/data/DUTS/DUTS-TR/gt/'
    test_root = test_path + '/imgs/'
    gt_test = test_path + '/gt/'
    train_loader,test_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,test_root=test_root,test_gt_root=gt_test)
    total_step = len(train_loader)
    
    test(test_loader,model,0,exp_id+1)