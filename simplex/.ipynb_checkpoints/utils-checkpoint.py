import torch
import time
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # print(tensor.numel())
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)

def assign_pars(vector, model):
    new_pars = unflatten_like(vector, model.parameters())
    for old, new in zip(model.parameters(), new_pars):
        old.data = new.to(old.device).data
    
    return