import torch
import src
from src import Metrics, utils
import collections
import os
import numpy as np
from PIL import Image
class Eval:
    def __init__(self, L):
        self.Loader = L
        self.scores = collections.OrderedDict()
        for val in L.vals:
            self.scores[val] = src.Score(val, L)

    def eval_Saliency(self, Model, epoch,exp_id, supervised=True,):
        savedict = {}
        # print(valdata['X'].shape)
        Outputs = {val : Metrics.getOutPuts(Model, valdata['X'], self.Loader, supervised=supervised) for val, valdata in self.Loader.valdatas.items()}

        # print(Outputs.keys())
        for val in self.Loader.valdatas.keys():
            Outputs[val]['Name'] = self.Loader.valdatas[val]['Name']
            Outputs[val]['Shape'] = self.Loader.valdatas[val]['Shape']
            Outputs[val]['gt'] = self.Loader.valdatas[val]['Y']

        for valname, output in Outputs.items():
            # print(valname)
            # print(val)
            save = 'output'
            save_1 = 'output/'+valname

            if not os.path.exists(save):
                os.makedirs(save)
            if not os.path.exists(save_1):
                os.makedirs(save_1)
                os.makedirs(save_1+'/pred')
                os.makedirs(save_1+'/gt')
            names, shapes, finals, time,gts = output['Name']['Y'], output['Shape'], output['final'] * 255., output['time'], output['gt']*255
            for i in range(len(names)):
                pred_p = 'output/'+valname+'/pred/'+names[i]
                gt_p = 'output/'+valname+'/gt/'+names[i]
                Image.fromarray(np.uint8(finals[i])).resize((shapes[i]), Image.BICUBIC).save(pred_p)
                Image.fromarray(np.uint8(gts[i])).resize((shapes[i]), Image.BICUBIC).save(gt_p)
            # print(F)


        #     saves = self.scores[val].update([F, M], epoch)
        #     savedict[val] = saves

        # for val, score in self.scores.items():
        #     score.print_present()
        # print('-----------------------------------------')

        # if self.Loader.MODE == 'train':
        #     torch.save(utils.makeDict(Model.state_dict()), utils.genPath(self.Loader.spath, 'present.pkl'))
        #     for val, saves in savedict.items():
        #         for idx, save in enumerate(saves):
        #             if save:
        #                 torch.save(utils.makeDict(Model.state_dict()), utils.genPath(self.Loader.spath, val+'_'+['F', 'M'][idx]+'.pkl'))
            
        #     for val, score in self.scores.items():
        #         score.print_best()

        # else:
        #     for val in self.Loader.valdatas.keys():
        #         Outputs[val]['Name'] = self.Loader.valdatas[val]['Name']
        #         Outputs[val]['Shape'] = self.Loader.valdatas[val]['Shape']

        #     return Outputs if self.Loader.save else None
