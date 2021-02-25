import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import argparse

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import glob

import tabulate

import sys
sys.path.append("../../simplex/")
import utils
from simplex_helpers import volume_loss
import surfaces
import time
sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16


def main(args):
    trial_num = len(glob.glob("./saved-outputs/model_*"))
    savedir = "./saved-outputs/model_" + str(trial_num) + "/"
    os.makedirs(savedir, exist_ok=True)
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset = torchvision.datasets.CIFAR10(args.data_path, 
                                           train=True, download=False,
                                           transform=transform_train)
    trainloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    
    testset = torchvision.datasets.CIFAR10(args.data_path, 
                                           train=False, download=False,
                                           transform=transform_test)
    testloader = DataLoader(testset, shuffle=True, batch_size=args.batch_size)
     
    model = VGG16(10)
    model = model.cuda()
    
    ## training setup ##
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr_init,
        momentum=0.9,
        weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()
    
    ## train ##
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time']
    for epoch in range(args.epochs):
        time_ep = time.time()
        train_res = utils.train_epoch(trainloader, model, criterion, optimizer)
        
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            test_res = utils.eval(testloader, model, criterion)
        else:
            test_res = {'loss': None, 'accuracy': None}
        
        time_ep = time.time() - time_ep
        
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], 
                  test_res['loss'], test_res['accuracy'], time_ep]

        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table, flush=True)

    checkpoint = model.state_dict()
    trial_num = len(glob.glob("./saved-outputs/model_*"))
    savedir = "./saved-outputs/model_" +\
               str(trial_num) + "/"
    os.makedirs(savedir, exist_ok=True)
    torch.save(checkpoint, savedir + "base_model.pt")

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="cifar10 simplex")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 50)",
    )

    parser.add_argument(
        "--lr_init",
        type=float,
        default=0.05,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--data_path",
        default="/datasets/",
        help="directory where datasets are stored",
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=5e-4,
        metavar="weight_decay",
        help="weight decay",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        metavar="epochs",
        help="number of training epochs",
    )
    parser.add_argument(
        '--eval_freq', 
        type=int, 
        default=5, 
        metavar='N', 
        help='evaluation frequency (default: 5)'
    )
    args = parser.parse_args()

    main(args)