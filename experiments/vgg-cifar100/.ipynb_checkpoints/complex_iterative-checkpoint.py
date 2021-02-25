import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import argparse
import time
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import sys
sys.path.append("../../simplex/")
import utils

sys.path.append("../../simplex/models/")
from vgg_noBN import VGG16, VGG16Simplex
from simplex_models import SimplexNet, Simplex

import tabulate
        
def main(args):
    
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
    dataset = torchvision.datasets.CIFAR100(args.data_path, 
                                           train=True, download=False,
                                           transform=transform_train)
    trainloader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size,
                            num_workers=4, pin_memory=True)
    testset = torchvision.datasets.CIFAR100(args.data_path, 
                                           train=False, download=False,
                                           transform=transform_test)
    testloader = DataLoader(testset, shuffle=True, batch_size=args.batch_size,
                            num_workers=4, pin_memory=True)
    
    fix_pts = [True] * args.n_mode
    n_vert = len(fix_pts)
    complex_ = {ii:[ii] for ii in range(args.n_mode)}
    simplex_model = SimplexNet(100, VGG16Simplex, n_vert=n_vert,
                               simplicial_complex=complex_,
                                fix_points=fix_pts).cuda()
    
    base_model = VGG16(100)
    for ii in range(args.n_mode):
        fname = "./saved-outputs/model_" + str(ii) + "/base_model.pt"
        base_model.load_state_dict(torch.load(fname))
        simplex_model.import_base_parameters(base_model, ii)
    
    
    ## add a new points and train ##
    for vv in range(args.n_connector):
        simplex_model.add_vert(to_simplexes=[ii for ii in range(args.n_mode)])
        simplex_model = simplex_model.cuda()
        optimizer = torch.optim.SGD(
            simplex_model.parameters(),
            lr=args.lr_init,
            momentum=0.9,
            weight_decay=args.wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=args.epochs)
        criterion = torch.nn.CrossEntropyLoss()
        columns = ['vert', 'ep', 'lr', 'tr_loss', 
                   'tr_acc', 'te_loss', 'te_acc', 'time', "vol"]
        
        print(simplex_model.simplicial_complex, flush=True)
        for epoch in range(args.epochs):
            time_ep = time.time()
            if v == 0:
                train_res = utils.train_epoch_multi_sample(trainloader, simplex_model, 
                                                     criterion, optimizer, args.n_sample)
            else:
                train_res = utils.train_epoch_volume(trainloader, simplex_model, 
                                                 criterion, optimizer, 
                                                 1e-10, args.n_sample)

            start_ep = (epoch == 0)
            eval_ep = epoch % args.eval_freq == args.eval_freq - 1
            end_ep = epoch == args.epochs - 1
            if start_ep or eval_ep or end_ep:
                test_res = utils.eval(testloader, simplex_model, criterion)
            else:
                test_res = {'loss': None, 'accuracy': None}

            time_ep = time.time() - time_ep

            lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            values = [vv, epoch + 1, lr, 
                      train_res['loss'], train_res['accuracy'], 
                      test_res['loss'], test_res['accuracy'], time_ep,
                     simplex_model.total_volume().item()]

            table = tabulate.tabulate([values], columns, 
                                      tablefmt='simple', floatfmt='8.4f')
            if epoch % 40 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table, flush=True)
            
        checkpoint = simplex_model.state_dict()
        fname = "./saved-outputs/" + str(args.n_mode) +\
                "mode_" + str(vv+1) + "connector.pt" 
        torch.save(checkpoint, fname)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="cifar100 simplex")

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
        default=0.01,
        metavar="LR",
        help="initial learning rate (default: 0.1)",
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=5e-4,
        metavar="weight_decay",
        help="weight decay",
    )
    
    parser.add_argument(
        "--data_path",
        default="/datasets/",
        help="directory where datasets are stored",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--n_mode",
        type=int,
        default=4,
        metavar="N",
        help="number of modes to connect",
    )

    parser.add_argument(
        "--n_connector",
        type=int,
        default=3,
        metavar="N",
        help="number of connecting points to use",
    )

    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10,
        metavar="N",
        help="how freq to eval test",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5,
        metavar="N",
        help="number of samples to use per iteration",
    )
    args = parser.parse_args()

    main(args)