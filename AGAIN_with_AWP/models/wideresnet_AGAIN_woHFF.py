# -*- coding: utf-8 -*-
'''
The code has been modified according to CAS
https://github.com/bymavis/CAS_ICLR2021
'''

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wide_resnet import BasicBlock, NetworkBlock

def func1(amount,num):
    list1 = []
    for i in range(0,num-1):
        a = random.uniform(0,amount)  
        list1.append(a)
    list1.sort()                       
    list1.append(amount)               

    list2 = []
    for i in range(len(list1)):
        if i == 0:
            b = list1[i]               
        else:
            b = list1[i] - list1[i-1]  
        list2.append(b)
    return list2

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adain(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def HFF(content,style):
    lam_mix = func1(1,4)
    size = content.size()
    out = adain(content,style)
    out2 = adain(style,content)
    
    style_mean, style_std = calc_mean_std(style)
    content_mean, content_std = calc_mean_std(content)
    out_mean, out_std = calc_mean_std(out)
    out2_mean, out2_std = calc_mean_std(out2)
    
    x_mix_std = lam_mix[0] * content_std + lam_mix[1] * style_std + lam_mix[2] * out_std + lam_mix[3] * out2_std
    x_mix_mean = lam_mix[0] * content_mean + lam_mix[1] * style_mean + lam_mix[2] * out_mean + lam_mix[3] * out2_mean
    
    normalized_feat = (content - content_mean.expand(
        size)) / content_std.expand(size)
    
    return normalized_feat * x_mix_std.expand(size) + x_mix_mean.expand(size)



class BasicBlock_ASE(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock_ASE, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.fc = nn.Linear(in_planes, 10)

    def forward(self, x, label=None):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        input = out if self.equalInOut else x
        fc_in = torch.mean(input.view(input.shape[0], input.shape[1], -1), dim=-1)
        fc_out = self.fc(fc_in.view(input.shape[0], input.shape[1]))
        if self.training:
            fake_label = label[torch.randperm(label.size(0))]
            N, C, H, W = input.shape
            mask = self.fc.weight[label, :]
            mask2 = self.fc.weight[fake_label, :]
            input = input * (0.6*mask.view(N, C, 1, 1) + 0.4*mask2.view(N, C, 1, 1))
        else:
            N, C, H, W = input.shape
            pred_label = torch.max(fc_out, dim=1)[1]
            mask = self.fc.weight[pred_label, :]
            input = input * mask.view(N, C, 1, 1)

        out = self.relu2(self.bn2(self.conv1(input)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        return out, fc_out
    

class NetworkBlock_ASE(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock_ASE, self).__init__()
        self.nb_layers = nb_layers
        self.layer = self._make_layer(BasicBlock_ASE, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        # for i in range(int(nb_layers)):
        #     layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        for i in range(int(nb_layers)-1):
            layers.append(BasicBlock(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        for i in range(int(nb_layers)-1, int(nb_layers)):
            layers.append(BasicBlock_ASE(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.ModuleList(layers)

    def forward(self, x, y=None):
        extra_output = []
        out = x
        for i in range(int(self.nb_layers)-1):
            out = self.layer[i](out)
        for i in range(int(self.nb_layers)-1, int(self.nb_layers)):
            out, fc_output = self.layer[i](out, y)
            extra_output.append(fc_output)

        return out, extra_output



class WideResNet(nn.Module):
    def __init__(self, depth=34, num_classes=10, widen_factor=10, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 1st sub-block
        self.sub_block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock_ASE(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.extra_fc = nn.Linear(nChannels[3], 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, y=None):
        extra_output = []
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        # out = self.block3(out)
        out, extra_output = self.block3(out, y)
        out = self.relu(self.bn1(out))

        fc_in = torch.mean(out.view(out.shape[0], out.shape[1], -1), dim=-1)
        fc_out = self.extra_fc(fc_in.view(out.shape[0], out.shape[1]))
        extra_output.append(fc_out)
        if self.training:
            fake_label = y[torch.randperm(y.size(0))]
            N, C, _, _ = out.shape
            mask = self.extra_fc.weight[y, :]
            mask2 = self.extra_fc.weight[fake_label, :]
            out = out * (0.6*mask.view(N, C, 1, 1) + 0.4*mask2.view(N, C, 1, 1))
        else:
            N, C, _, _ = out.shape
            pred_label = torch.max(fc_out, dim=1)[1]
            mask = self.extra_fc.weight[pred_label, :]
            out = out * mask.view(N, C, 1, 1)

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), extra_output

def WideResNet34(num_classes=10):
    return WideResNet(depth=34, num_classes=num_classes)
