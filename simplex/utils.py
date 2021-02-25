import torch
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


def eval(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        with torch.no_grad():
            output = model(input_var)
            # print(output)
            # output = output
            loss = criterion(output, target_var)

        loss_sum += loss.data.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }

def train_epoch(loader, model, criterion, optimizer):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def train_epoch_volume(loader, model, criterion, optimizer, vol_reg,
                      nsample):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        acc_loss = 0.
        for _ in range(nsample):
                output = model(input_var)
                acc_loss = acc_loss + criterion(output, target_var)
        acc_loss.div(nsample)
        
        vol = model.total_volume()
        log_vol = (vol + 1e-4).log()
        
        loss = acc_loss - vol_reg * log_vol

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def train_epoch_multi_sample(loader, model, criterion, 
                             optimizer, nsample):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        acc_loss = 0.
        for _ in range(nsample):
                output = model(input_var)
                acc_loss += criterion(output, target_var)
        acc_loss.div(nsample)
        
        loss = acc_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def train_transformer_epoch(
        loader, model, criterion, optimizer, nsample, vol_reg=1e-5, gradient_accumulation_steps=1
):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        if i % 20 == 0:
            print(i, "batches completed")
        torch.cuda.empty_cache()

        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        loss = 0.
        for j in range(nsample):
            output = model(input_var)[0]
            loss += criterion(output, target_var)
        loss.div(nsample)
        
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        vol = model.total_volume()
        log_vol = vol_reg * (vol + 1e-4).log()
        loss = loss - log_vol

        # optimizer.zero_grad()
        loss.backward()
        
        if (i + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }
