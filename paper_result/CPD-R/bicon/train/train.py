"""
BiconNet: An Edge-preserved Connectivity-based Approach for Salient Object Detection
Ziyun Yang, Somayyeh Soltanian-Zadeh and Sina Farsiu
Codes from: https://github.com/Zyun-Y/BiconNets
Paper: https://arxiv.org/abs/2103.00334
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime
from connect_loss import bicon_loss
from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from data import get_loader
from utils import clip_gradient, adjust_lr
from salient_eva import Eval_thread
from utils_bicon import *
torch.cuda.set_device(0)
save = 'save'
if not os.path.exists(save):
    os.makedirs(save)


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=80, help='epoch number')
parser.add_argument('--lr', type=float, default=3.5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=True, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.2, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=45, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))
# build models
if opt.is_ResNet:
    model = CPD_ResNet()
else:
    model = CPD_VGG()

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)


image_root = '/data/DUTS/DUTS-TR/imgs/'
gt_root = '/data/DUTS/DUTS-TR/gt/'
test_root= '/data/ECSSD/imgs/'
test_gt_root = '/data/ECSSD/gt/'
train_loader,test_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,test_root=test_root,test_gt_root=test_gt_root)

total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
best_mae = 1
# training

csv = 'results.csv'
with open(os.path.join(save, csv), 'w') as f:
    f.write('epoch, mae, maxF\n')

def train(train_loader, model, optimizer, epoch):

    model.train()
    loss_func = bicon_loss()
    total_loss = 0
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, conn = pack
        images = Variable(images)
        gts = Variable(gts)
        conn = Variable(conn)
        images = images.cuda()
        gts = gts.cuda()
        conn = conn.cuda()

        atts, dets = model(images)
        loss = loss_func(atts,dets,gts,conn)

        loss.backward()
        total_loss += loss.data
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        if i % 400 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f} '.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, total_loss/i))

    

    if opt.is_ResNet:
        save_path = 'models/CPD_Resnet/'
    else:
        save_path = 'models/CPD_VGG/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model.state_dict(), save_path + 'CPD.pth' + '.%d' % epoch)

def test(test_loader,model,epoch):
        model.eval()
        Eval = Eval_thread()
        n=0
        mae_ls = []
        fmax_ls = []

        with torch.no_grad(): 
            for j_batch, test_data in enumerate(test_loader):
                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])
                X_test= X_test.cuda()
                y_test = y_test.cuda()
                # print(X_test.shape,y_test.shape)
                if X_test.shape[1]==1:
                    X_test = torch.cat([X_test,X_test,X_test],1)
                # print(X_test.shape,y_test.shape)
                _, output_test = model(X_test)


                pred =bv_test(output_test)
                
                pred = F.upsample(pred, size=(y_test.shape[2],y_test.shape[3]), mode='bilinear', align_corners=False)
                
                for im in range(y_test.shape[0]):

                    mae, fmax  = Eval.run(pred[im],y_test[im])
                    mae_ls.append(mae)
                    fmax_ls.append(fmax)
                # print(j_batch)
                if j_batch % 100==0:
                    print('test [Iteration : ' + str(j_batch) + '/' + str(len(test_loader)) + '] fmax:%.3f' ' mae:%.3f' %(
                        np.mean(fmax_ls),np.mean(mae_ls)))

            csv = 'results.csv'
            with open(os.path.join('save', csv), 'a') as f:
                f.write('%03d,%0.6f,%0.5f \n' % (
                    (epoch + 1),
                    np.mean(mae_ls),
                    np.mean(fmax_ls)
                ))
            return np.mean(mae_ls)



print("Let's go!")
for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
    if epoch %2==0:
        curr_mae = test(test_loader,model,epoch)
        if curr_mae < best_mae:
            best_mae = curr_mae
            torch.save(model.state_dict(), 'models/CPD_Resnet/best_model.pth')

