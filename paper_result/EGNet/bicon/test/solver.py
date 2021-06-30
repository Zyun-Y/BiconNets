from random import shuffle
import sys
import datetime
import numpy as np
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import os
from tensorboardX import SummaryWriter
import torchvision.utils as utils
from skimage.io import imread, imsave
from lib.bad_grad_viz import register_hooks
from apex import amp
from lr_update import get_lr
from utils_bicon import *
save = 'save'
if not os.path.exists(save):
    os.makedirs(save)




class Solver(object):

    def train(self, model, train_loader, val_loader,cfg):
        """
        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        """
        exp_id = cfg.exp_id

        model.cuda()

        print('START TRAIN.')

        csv = 'results_'+str(exp_id)+'.csv'
        with open(os.path.join(save, csv), 'w') as f:
            f.write('epoch, mae\n')

        self.test_epoch(model,val_loader,-1,exp_id)
        
    def test_epoch(self,model,loader,epoch,exp_id):
        mae_ls_binary = []
        step = 0
        model.eval()
        with torch.no_grad(): 
            for j_batch, data_batch in enumerate(loader):
                curr_dice = []
                step+=1
                X_test,y_test, name, im_size = data_batch['image'],data_batch['label'], data_batch['name'][0], np.asarray(data_batch['size'])

                X_test= X_test.cuda()

                y_test = y_test.long().cuda()

                up_edge, up_sal, final_sal = model(X_test)
                output_test = final_sal[-1]


                pred =bv_test(output_test)

                for im in range(y_test.shape[0]):
                    # n = n+1
                    save_fig(pred[im][0],y_test[im][0],exp_id,name)

                    mae_bi = Eval_mae(pred[im],y_test[im])
                    mae_ls_binary.append(mae_bi)

                if step%100 == 0:
                    print('test [Iteration : ' + str(j_batch) + '/' + str(len(loader)) + ']' ' ave mae:%.3f'%(np.mean(mae_ls_binary)))

            csv = 'results_'+str(exp_id)+'.csv'
            with open(os.path.join(save, csv), 'a') as f:
                f.write('%03d,%0.6f \n' % (
                    (epoch + 1),
                    np.mean(mae_ls_binary)
                ))

            return np.mean(mae_ls_binary)

def Eval_mae(pred,gt):
    avg_mae, img_num = 0.0, 0.0
    #mae_list = [] # for debug
    with torch.no_grad():
        mea = torch.abs(pred - gt).mean()

    return mea.item()

def save_fig(pred,y_test,exp_id,name):
    path = 'output/'+str(exp_id)
    path1 = 'output/'+str(exp_id)+'/gt'
    path2 = 'output/'+str(exp_id)+'/pred'
    if not os.path.exists(path1):
        os.makedirs(path)
        os.makedirs(path1)
        os.makedirs(path2)

    
    imsave('output/'+str(exp_id)+'/gt/'+name+'.png',(y_test*255.0).cpu().numpy().astype(np.uint16))
    imsave('output/'+str(exp_id)+'/pred/'+name+'.png',(pred*255.0).squeeze().cpu().numpy().astype(np.uint16))