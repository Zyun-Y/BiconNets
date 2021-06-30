import argparse
import os
from dataset import get_loader
from solver import Solver
import torch
torch.cuda.set_device(0)

def main(config):
    if config.mode == 'train':
        train_loader, _ = get_loader(config, mode='train')
        test_loader, _ = get_loader(config, mode='test')
        run = "nnet"
        if not os.path.exists("%s/run-%s" % (config.save_fold, run)): 
            os.mkdir("%s/run-%s" % (config.save_fold, run))
            os.mkdir("%s/run-%s/logs" % (config.save_fold, run))
            os.mkdir("%s/run-%s/models" % (config.save_fold, run))
        config.save_fold = "%s/run-%s" % (config.save_fold, run)
        train = Solver(train_loader,test_loader, config)
        train.train()
    elif config.mode == 'test':
        test_loader, dataset = get_loader(config, mode='test')

        test = Solver(None, test_loader, config, dataset.save_folder())
        test.test(test_mode=config.test_mode)
    else:
        raise IOError("illegal input!!!")



if __name__ == '__main__':

    vgg_path = '/vgg16_20M.pth'
    resnet_path = '/github/resnet50_caffe.pth'

    train_root = '/data/DUTS/DUTS-TR'
    test_root = '/data/ECSSD'
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)

    parser.add_argument('--cuda', type=bool, default=True)

    # Training settings
    parser.add_argument('--train_root', type=str, default=train_root)
    parser.add_argument('--test_root', type=str, default=test_root)

    parser.add_argument('--vgg', type=str, default=vgg_path)
    parser.add_argument('--resnet', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=27) # 12, now x3
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=4)
    parser.add_argument('--load_bone', type=str, default='')
    # parser.add_argument('--load_branch', type=str, default='')
    parser.add_argument('--save_fold', type=str, default='./EGNet')
    # parser.add_argument('--epoch_val', type=int, default=20)
    parser.add_argument('--epoch_save', type=int, default=3) # 2, now x3
    parser.add_argument('--epoch_show', type=int, default=1)
    parser.add_argument('--pre_trained', type=str, default=None)

    # Testing settings
    parser.add_argument('--model', type=str, default='./epoch_resnet.pth')
    parser.add_argument('--test_fold', type=str, default='output')
    parser.add_argument('--test_mode', type=int, default=1)
    parser.add_argument('--sal_mode', type=str, default='t')

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--visdom', type=bool, default=False)
    
    config = parser.parse_args()
  
    if not os.path.exists(config.save_fold): os.mkdir(config.save_fold)
    main(config)
