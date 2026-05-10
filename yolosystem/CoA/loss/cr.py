import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models
from torchvision.models import VGG19_Weights


class Vgg19(torch.nn.Module):
    #新增device参数
    def __init__(self, requires_grad=False,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


# class ContrastLoss(nn.Module):
#     #源代码 def __init__(self, ablation=False):
#     def __init__(self, ablation=False, device='cpu'):
#         super(ContrastLoss, self).__init__()
#         #源代码 self.vgg = Vgg19().cuda()
#         self.vgg = Vgg19().to(device)  # 而不是硬编码的 .cuda()
#         self.l1 = nn.L1Loss()
#         self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
#         self.ab = ablation
#         self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
#         self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()
class ContrastLoss(nn.Module):
    #这里将默认cpu修改成使用gpu
    def __init__(self, ablation=False, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().to(device)  # 使用传入的device参数
        self.l1 = nn.L1Loss().to(device)  # 损失函数也移到指定设备
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation
        # 修改这两个张量的设备指定方式
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(device)

    def forward(self, a, p, n):
        print(f"Input devices - a: {a.device}, p: {p.device}, n: {n.device}")
        print(f"Model devices - mean: {self.mean.device}, std: {self.std.device}")
        print(f"VGG devices - first weight: {next(self.vgg.parameters()).device}")
        #a-anchor, p-positive, n-negative
        a = a.to(self.mean.device)  # 确保 a 和 mean/std 在同一个设备上
        # 确保 p 和 n 也在同一个设备上
        p = p.to(self.mean.device)
        n = n.to(self.mean.device)
            # 归一化
        a = (a - self.mean) / self.std
        p = (p - self.mean) / self.std
        n = (n - self.mean) / self.std
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss


class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()
        self.device = device  # 存储 device
        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(pretrained=True).features
        # self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
        # self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()
        #确保使用同一个设备
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(device)
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        # source = (source - self.mean) / self.std
        # target = (target - self.mean) / self.std
        source = source.to(self.mean.device)
        target = target.to(self.mean.device)
        #归一化
        source = (source - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss
