import torch
import torch.nn as nn
import torch.nn.functional as F

from simplex_models import SimplexNet, Simplex
from simplex_models import Linear as SimpLinear
from simplex_models import Conv2d as SimpConv
from simplex_models import BatchNorm2d as SimpBN

class Block(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1):
        super(Block, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, 
                     stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return self.relu(out)
    
class BasicConv(nn.Module):
    """
    Very small CNN
    """
    def __init__(self, num_classes, k=32):
        super(BasicConv, self).__init__()
        self.num_classes = num_classes
        self.l1 = Block(3,k)
        self.l2 = Block(k,k)
        self.l3 = Block(k,2*k)
        self.maxpool = nn.MaxPool2d(2)
        
        self.l4 = Block(2*k,2*k)
        self.l5 = Block(2*k,2*k)
        self.l6 = Block(2*k,2*k)
        self.l7 = Block(2*k,2*k)
        self.expression = Expression(lambda u:u.mean(-1).mean(-1))
        self.fc = nn.Linear(2*k, num_classes)
    def forward(self,x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.maxpool(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.expression(x)
        x = self.fc(x)
        return x
    
    
class SimplexBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride=1, fix_points=None):
        super(SimplexBlock, self).__init__() 
        self.conv = SimpConv(in_channels, out_channels, 3, padding=1, 
                     stride=stride,
                     fix_points=fix_points)
        self.bn = SimpBN(out_channels, fix_points=fix_points)
        self.relu = nn.ReLU()
    def forward(self, x, coeffs_t):
        out = self.conv(x, coeffs_t)
        out = self.bn(out, coeffs_t)
        return self.relu(out)
    
class SimplexConv(nn.Module):
    """
    Very small CNN
    """
    def __init__(self, num_classes, fix_points, k=32):
        super(SimplexConv, self).__init__()
        self.num_classes = num_classes
        self.l1 = SimplexBlock(3,k, fix_points=fix_points)
        self.l2 = SimplexBlock(k,k, fix_points=fix_points)
        self.l3 = SimplexBlock(k,2*k, fix_points=fix_points)
        self.maxpool = nn.MaxPool2d(2)
        
        self.l4 = SimplexBlock(2*k,2*k, fix_points=fix_points)
        self.l5 = SimplexBlock(2*k,2*k, fix_points=fix_points)
        self.l6 = SimplexBlock(2*k,2*k, fix_points=fix_points)
        self.l7 = SimplexBlock(2*k,2*k, fix_points=fix_points)
        self.expression = Expression(lambda u:u.mean(-1).mean(-1))
        self.fc = SimpLinear(2*k,num_classes, fix_points=fix_points)
    def forward(self,x, coeffs_t):
        x = self.l1(x, coeffs_t)
        x = self.l2(x, coeffs_t)
        x = self.l3(x, coeffs_t)
        x = self.maxpool(x)
        x = self.l4(x, coeffs_t)
        x = self.l5(x, coeffs_t)
        x = self.l6(x, coeffs_t)
        x = self.l6(x, coeffs_t)
        x = self.l7(x, coeffs_t)
        x = self.expression(x)
        x = self.fc(x, coeffs_t)
        return x

class Expression(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
    
    