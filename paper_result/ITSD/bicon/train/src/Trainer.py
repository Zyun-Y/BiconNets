import time
import torch
import src
import datetime
from src import utils
from progress.bar import Bar
from src.Loss import bicon_loss
from torch import nn
import os
from utils_bicon import *

def freeze_bn(model):
    for m in model.encoder.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False
    
                    
class Trainer(src.Experiment):
    def __init__(self, L, E):
        super(Trainer, self).__init__(L, E)
        self.epochs = L.iter // L.epoch
        multi_gpu = len(L.ids) > 1
        
        self.params = utils.genParams(L.plist, model=self.Model.module)
        self.optimizer = self.optims(L.optim, self.params)
        if multi_gpu:
            self.optimizer = torch.nn.DataParallel(self.optimizer, device_ids=L.ids)
            self.optimizer = self.optimizer.module
        freeze_bn(self.Model.module)
        self.scheduler = self.schedulers(L.scheduler, self.optimizer) if L.scheduler != 'None' else None
        self.loss = bicon_loss
        self.best_mae = 1

    def epoch(self, idx):
        # curr_mae = self.Eval.eval_Saliency(self.Model, epoch=idx, supervised=False)
        st = time.time()
        ans = 0
        print('---------------------------------------------------------------------------')
        # bar = Bar('{} | epoch {}:'.format(self.Loader.sub, idx), max=self.Loader.epoch)
        # curr_mae = self.Eval.eval_Saliency(self.Model, epoch=idx, supervised=False)
        for i in range(self.Loader.epoch):
            self.optimizer.zero_grad()
            batchs = self.Loader.trSet.getBatch(self.Loader.batch)
            X = torch.tensor(batchs['X'], requires_grad=True).float().cuda(self.Loader.ids[0])
            _y = self.Model(X, 'tr')
            loss = self.loss(_y, batchs, self.Loader)
            X, _y = 0, 0
            ans += loss.cpu().data.numpy()
            loss.backward()
            if i%100==0:
                print(print('%s | step:%d/%d/%d/%d  | loss=%.6f'%(datetime.datetime.now(), i,self.Loader.epoch, idx,self.epochs, ans/100)))
                ans=0
            # Bar.suffix = '{}/{} | loss: {}'.format(i, self.Loader.epoch, ans * 1. / (i + 1))
            self.optimizer.step()
            # bar.next()

        # bar.finish()

        print('epoch: {},  time: {}, loss: {:.5f}.'.format(idx, time.time() - st, ans * 1. / self.Loader.epoch))

        st = time.time()
        curr_mae = self.Eval.eval_Saliency(self.Model, epoch=idx, supervised=False)
        if curr_mae<self.best_mae:
            self.best_mae = curr_mae
            torch.save(self.Model.state_dict(), 'save/best_model.pth')
        print('Evaluate using time: {:.5f}.'.format(time.time() - st))

    def train(self):
        save = 'save'
        if not os.path.exists(save):
            os.makedirs(save)

        csv = 'results.csv'
        with open(os.path.join(save, csv), 'w') as f:
            f.write('epoch, Fmax, MAE\n')

        for idx in range(self.epochs):
            self.scheduler.step()
            self.epoch(idx+1)
            if idx >15:
                torch.save(self.Model.state_dict(), 'save/'+'model_'+str(idx)+'.pth')