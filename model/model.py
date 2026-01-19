import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models
from torch.autograd import Function
from torch.autograd import Variable
from .model_util import *


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)



resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

class GradientReversalLayer(Function):
    """梯度反转层 (Forward不变，Backward反转梯度)"""
    @staticmethod
    def forward(ctx, x, lamda):
        alpha = 1.0+1.0 * (1 - lamda)
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.alpha, None

class FeatureExtractor(nn.Module):
    def __init__(self, resnet_name="ResNet18"):
        super(FeatureExtractor, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        class_num = 241
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.layer0 = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        self.feature_layers = nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)

        self.__in_features = model_resnet.fc.in_features

    def forward(self, x,half=False):
        if half:
            y = self.avgpool(x)
            y = y.view(y.size(0), -1)
            return y
        else:
            low = self.layer0(x)
            rec = self.feature_layers(low)
            y = self.avgpool(rec)
            y = y.view(y.size(0), -1)
            return rec,y

    def get_1x_lr_params_NOscale(self):
        b = []

        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        # b.append(self.bottleneck.parameters())
        b.append(self.fc.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr':  1* learning_rate}]
                # {'params': self.get_10x_lr_params(),        'lr': 10* learning_rate}]

    def output_num(self):
      return self.__in_features

class TaskClassifier(nn.Module):
    """任务分类器 (预测类别标签)"""
    def __init__(self, num_classes):
        super().__init__()
        model_resnet = resnet_dict["ResNet18"](pretrained=True)
        self.fc = nn.Linear(model_resnet.fc.in_features, num_classes)
        self.fc.apply(init_weights)
    
    def forward(self, y):
        y1 = self.fc(y)
        return y1

class DomainClassifier(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(DomainClassifier, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        # self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.grl = GradientReversalLayer.apply

    def forward(self, x,lamda):
        x = self.grl(x, lamda)  # 应用梯度反转
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y2 = F.log_softmax(y, dim=1)
        return y2

    def output_num(self):
        return 2
    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]



        
