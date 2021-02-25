import torch
import torch.nn as nn
from simplex_models import Linear as SimpLinear
from simplex_models import Conv2d as SimpConv
from simplex_models import BatchNorm2d as SimpBN
import math


class VGG16(nn.Module):
    def __init__(self, n_classes=10):
        super(VGG16, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.c4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.c6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.c7 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c8 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.c9 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn9 = nn.BatchNorm2d(512)
        self.c10 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn10 = nn.BatchNorm2d(512)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c11 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(512)
        self.c12 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(512)
        self.c13 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(512)
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.d1 = nn.Dropout()
        self.l1 = nn.Linear(512, 512)
        self.d2 = nn.Dropout()
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, n_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu(self.bn1(self.c1(x)))
        x = self.relu(self.bn2(self.c2(x)))
        x = self.mp1(x)
        
        x = self.relu(self.bn3(self.c3(x)))
        x = self.relu(self.bn4(self.c4(x)))
        x = self.mp2(x)
        
        x = self.relu(self.bn5(self.c5(x)))
        x = self.relu(self.bn6(self.c6(x)))
        x = self.relu(self.bn7(self.c7(x)))
        x = self.mp3(x)
        
        x = self.relu(self.bn8(self.c8(x)))
        x = self.relu(self.bn9(self.c9(x)))
        x = self.relu(self.bn10(self.c10(x)))
        x = self.mp4(x)
        
        x = self.relu(self.bn11(self.c11(x)))
        x = self.relu(self.bn12(self.c12(x)))
        x = self.relu(self.bn13(self.c13(x)))
        x = self.mp5(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.d1(x)
        x = self.relu(self.l1(x))
        x = self.d2(x)
        x = self.relu(self.l2(x))
        x = self.l3(x)
        
        return x
        
class VGG16Simplex(nn.Module):
    def __init__(self, n_classes=10, fix_points=[False]):
        super(VGG16Simplex, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        self.c1 = SimpConv(3, 64, 3, padding=1,
                             fix_points=fix_points)
        self.bn1 = SimpBN(64, fix_points=fix_points)
        self.c2 = SimpConv(64, 64, 3, padding=1,
                             fix_points=fix_points)
        self.bn2 = SimpBN(64, fix_points=fix_points)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c3 = SimpConv(64, 128, 3, padding=1,
                             fix_points=fix_points)
        self.bn3 = SimpBN(128, fix_points=fix_points)
        self.c4 = SimpConv(128, 128, 3, padding=1,
                             fix_points=fix_points)
        self.bn4 = SimpBN(128, fix_points=fix_points)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c5 = SimpConv(128, 256, 3, padding=1,
                             fix_points=fix_points)
        self.bn5 = SimpBN(256, fix_points=fix_points)
        self.c6 = SimpConv(256, 256, 3, padding=1,
                             fix_points=fix_points)
        self.bn6 = SimpBN(256, fix_points=fix_points)
        self.c7 = SimpConv(256, 256, 3, padding=1,
                             fix_points=fix_points)
        self.bn7 = SimpBN(256, fix_points=fix_points)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c8 = SimpConv(256, 512, 3, padding=1,
                             fix_points=fix_points)
        self.bn8 = SimpBN(512, fix_points=fix_points)
        self.c9 = SimpConv(512, 512, 3, padding=1,
                             fix_points=fix_points)
        self.bn9 = SimpBN(512, fix_points=fix_points)
        self.c10 = SimpConv(512, 512, 3, padding=1,
                             fix_points=fix_points)
        self.bn10 = SimpBN(512, fix_points=fix_points)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c11 = SimpConv(512, 512, 3, padding=1,
                             fix_points=fix_points)
        self.bn11 = SimpBN(512, fix_points=fix_points)
        self.c12 = SimpConv(512, 512, 3, padding=1,
                             fix_points=fix_points)
        self.bn12 = SimpBN(512, fix_points=fix_points)
        self.c13 = SimpConv(512, 512, 3, padding=1,
                             fix_points=fix_points)
        self.bn13 = SimpBN(512, fix_points=fix_points)
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.d1 = nn.Dropout()
        self.l1 = SimpLinear(512, 512, fix_points=fix_points)
        self.d2 = nn.Dropout()
        self.l2 = SimpLinear(512, 512, fix_points=fix_points)
        self.l3 = SimpLinear(512, n_classes, fix_points=fix_points)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x, coeffs_t):
        x = self.relu(self.bn1(self.c1(x, coeffs_t), coeffs_t))
        x = self.relu(self.bn2(self.c2(x, coeffs_t), coeffs_t))
        x = self.mp1(x)
        
        x = self.relu(self.bn3(self.c3(x, coeffs_t), coeffs_t))
        x = self.relu(self.bn4(self.c4(x, coeffs_t), coeffs_t))
        x = self.mp2(x)
        
        x = self.relu(self.bn5(self.c5(x, coeffs_t), coeffs_t))
        x = self.relu(self.bn6(self.c6(x, coeffs_t), coeffs_t))
        x = self.relu(self.bn7(self.c7(x, coeffs_t), coeffs_t))
        x = self.mp3(x)
        
        x = self.relu(self.bn8(self.c8(x, coeffs_t), coeffs_t))
        x = self.relu(self.bn9(self.c9(x, coeffs_t), coeffs_t))
        x = self.relu(self.bn10(self.c10(x, coeffs_t), coeffs_t))
        x = self.mp4(x)
        
        x = self.relu(self.bn11(self.c11(x, coeffs_t), coeffs_t))
        x = self.relu(self.bn12(self.c12(x, coeffs_t), coeffs_t))
        x = self.relu(self.bn13(self.c13(x, coeffs_t), coeffs_t))
        x = self.mp5(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.d1(x)
        x = self.relu(self.l1(x, coeffs_t))
        x = self.d2(x)
        x = self.relu(self.l2(x, coeffs_t))
        x = self.l3(x, coeffs_t)
        
        return x
