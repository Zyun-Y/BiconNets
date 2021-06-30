import pretrainedmodels
import torch.nn as nn
from torch.nn import functional as F
import torch
from unet_parts import *
import pretrainedmodels
# from models.base_model import *

import shutil
# from utils.util import *
from collections import OrderedDict
# from tensorboardX import SummaryWriter

# res = pretrainedmodels.resnet50(1000,pretrained=None)
# print(res)
affine_par = True
def load_pretrained_model(net, state_dict, strict=True):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True`` then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :func:`state_dict()` function.

    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        strict (bool): Strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's `:func:`state_dict()`
            function.
    """
    own_state = net.state_dict()
    # print state_dict.keys()
    # print own_state.keys()
    for name, param in state_dict.items():
        if name in own_state:
            # print name, np.mean(param.numpy())
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if strict:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            else:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    print('Ignoring Error: While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))

        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Bottleneck50(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(Bottleneck50, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out
class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                            padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out

class ResNet50(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        # self.conv2 = conv3x3(64, 64)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.relu2 = nn.ReLU(inplace=False)
        # self.conv3 = conv3x3(64, 128)
        # self.conv3 = DeformConv(64, 128, (3, 3), stride=1, padding=1, num_deformable_groups=1)
        # self.conv3_deform = conv3x3(64, 2 * 3 * 3)

        # self.bn3 = nn.BatchNorm2d(128)
        # self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        # input 528 * 528
        x_1 = self.relu1(self.bn1(self.conv1(x)))  # 264 * 264
        # x = self.relu2(self.bn2(self.conv2(x)))  # 264 * 264
        # x = self.relu3(self.bn3(self.conv3(x)))  # 264 * 264

        # x_13 = x
        # print(x.shape)
        x_2 = self.maxpool(x_1)  # 66 * 66
        x_2 = self.layer1(x_2)  # 66 * 66
        x_3 = self.layer2(x_2)  # 33 * 33
        x_4 = self.layer3(x_3)  # 66 * 66
        # x_46 = x
        x_5 = self.layer4(x_4)  # 33 * 33
        # print(x.shape)
        # x_13 = F.interpolate(x_13, [x_46.size()[2], x_46.size()[3]], mode='bilinear', align_corners=True)
        # x_low = torch.cat((x_13, x_46), dim=1)
        return x_1,x_2,x_3,x_4,x_5


class res34(nn.Module):
    def __init__(self):
        super(res34, self).__init__()
        res = pretrainedmodels.resnet34(1000)
        self.conv1 = res.conv1
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1= res.bn1
        self.relu = res.relu
        self.maxpool = res.maxpool

        self.layer1 = res.layer1
        self.layer2 = res.layer2
        self.layer3 = res.layer3
        self.layer4 = res.layer4

    def forward(self,x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.maxpool(x1)
        x2 = self.layer1(x2)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x1,x2,x3,x4,x5

class res_unet(nn.Module):
    def __init__(self, n_classes=8,pretrain=False, model_path=' ',bilinear=True):
        super(res_unet, self).__init__()
        self.model = res34()
        # if pretrain:
        #     load_pretrained_model(self.model, torch.load(model_path), strict=False)
        self.up1 = Up34(512, 256, bilinear)
        self.up2 = Up34(768, 128, bilinear)
        self.up3 = Up34(256, 64, bilinear)
        self.up4 = Up34(128, 64, bilinear)
        # self.up5 = Up34(128, 64, bilinear)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = OutConv(128, n_classes)
    def forward(self, x):
        x1,x2,x3,x4,x5 = self.model(x)
        # print(x4.shape)
        x = self.up(x5)
        x = torch.cat([x, x4], dim=1)
        # x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)

        x = self.up4(x, x1)
        x = self.up(x)
        # x = self.up5(x,None)
        logits = self.outc(x)
        # print(logits.shape)
        return logits
        # return x

# encoder = res_unet(pretrain=True, model_path='/home/ziyun/Desktop/Project/esophagus/code/github/resnet50-19c8e357.pth')
# print(encoder.model)