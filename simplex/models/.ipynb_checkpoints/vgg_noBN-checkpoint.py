import torch
import torch.nn as nn
from simplex_models import Linear as SimpLinear
from simplex_models import Conv2d as SimpConv
import math


class VGG16(nn.Module):
    def __init__(self, n_classes=10):
        super(VGG16, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        self.c1 = nn.Conv2d(3, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.c4 = nn.Conv2d(128, 128, 3, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c5 = nn.Conv2d(128, 256, 3, padding=1)
        self.c6 = nn.Conv2d(256, 256, 3, padding=1)
        self.c7 = nn.Conv2d(256, 256, 3, padding=1)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c8 = nn.Conv2d(256, 512, 3, padding=1)
        self.c9 = nn.Conv2d(512, 512, 3, padding=1)
        self.c10 = nn.Conv2d(512, 512, 3, padding=1)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c11 = nn.Conv2d(512, 512, 3, padding=1)
        self.c12 = nn.Conv2d(512, 512, 3, padding=1)
        self.c13 = nn.Conv2d(512, 512, 3, padding=1)
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
        x = self.relu(self.c1(x))
        x = self.relu(self.c2(x))
        x = self.mp1(x)
        
        x = self.relu(self.c3(x))
        x = self.relu(self.c4(x))
        x = self.mp2(x)
        
        x = self.relu(self.c5(x))
        x = self.relu(self.c6(x))
        x = self.relu(self.c7(x))
        x = self.mp3(x)
        
        x = self.relu(self.c8(x))
        x = self.relu(self.c9(x))
        x = self.relu(self.c10(x))
        x = self.mp4(x)
        
        x = self.relu(self.c11(x))
        x = self.relu(self.c12(x))
        x = self.relu(self.c13(x))
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
        self.c2 = SimpConv(64, 64, 3, padding=1,
                             fix_points=fix_points)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c3 = SimpConv(64, 128, 3, padding=1,
                             fix_points=fix_points)
        self.c4 = SimpConv(128, 128, 3, padding=1,
                             fix_points=fix_points)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c5 = SimpConv(128, 256, 3, padding=1,
                             fix_points=fix_points)
        self.c6 = SimpConv(256, 256, 3, padding=1,
                             fix_points=fix_points)
        self.c7 = SimpConv(256, 256, 3, padding=1,
                             fix_points=fix_points)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c8 = SimpConv(256, 512, 3, padding=1,
                             fix_points=fix_points)
        self.c9 = SimpConv(512, 512, 3, padding=1,
                             fix_points=fix_points)
        self.c10 = SimpConv(512, 512, 3, padding=1,
                             fix_points=fix_points)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.c11 = SimpConv(512, 512, 3, padding=1,
                             fix_points=fix_points)
        self.c12 = SimpConv(512, 512, 3, padding=1,
                             fix_points=fix_points)
        self.c13 = SimpConv(512, 512, 3, padding=1,
                             fix_points=fix_points)
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
        x = self.relu(self.c1(x, coeffs_t))
        x = self.relu(self.c2(x, coeffs_t))
        x = self.mp1(x)
        
        x = self.relu(self.c3(x, coeffs_t))
        x = self.relu(self.c4(x, coeffs_t))
        x = self.mp2(x)
        
        x = self.relu(self.c5(x, coeffs_t))
        x = self.relu(self.c6(x, coeffs_t))
        x = self.relu(self.c7(x, coeffs_t))
        x = self.mp3(x)
        
        x = self.relu(self.c8(x, coeffs_t))
        x = self.relu(self.c9(x, coeffs_t))
        x = self.relu(self.c10(x, coeffs_t))
        x = self.mp4(x)
        
        x = self.relu(self.c11(x, coeffs_t))
        x = self.relu(self.c12(x, coeffs_t))
        x = self.relu(self.c13(x, coeffs_t))
        x = self.mp5(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.d1(x)
        x = self.relu(self.l1(x, coeffs_t))
        x = self.d2(x)
        x = self.relu(self.l2(x, coeffs_t))
        x = self.l3(x, coeffs_t)
        
        return x