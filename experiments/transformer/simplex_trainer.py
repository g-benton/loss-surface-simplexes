import math
import torch
from torch import nn
import numpy as np
import pandas as pd
import argparse

from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import glob

import tabulate

import os
import sys

sys.path.append("../../simplex/")
import utils as simp_utils

import time
sys.path.append("../../simplex/models/")
from basic_simplex import BasicSimplex

sys.path.append("../../../ViT-pytorch/utils")
from data_utils import get_loader
from models.modeling import VisionTransformer, CONFIGS
        
def main(args):
    savedir = "./saved-outputs/model" + str(args.base_idx) + "/"
    print('Preparing directory %s' % savedir)
    os.makedirs(savedir, exist_ok=True)
    with open(os.path.join(savedir, 'base_command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
    
    trainloader, testloader = get_loader(args)
    
    config = CONFIGS['ViT-B_16']
    num_classes = 100
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    modeldir = "./cifar100-100_500_seed_" + str(args.base_idx) + "/"  
    modelname = "cifar100-100_500_seed_" + str(args.base_idx) + "_checkpoint.bin"
    model.load_state_dict(torch.load(modeldir+modelname))
    
    simplex_model = BasicSimplex(model, num_vertices=1, fixed_points=[False]).cuda()
    del model

    ## add a new points and train ##
    for vv in range(1, args.n_verts+1):
        simplex_model.add_vert()
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
                   'tr_acc', 'te_loss', 'te_acc', 'time']
        for epoch in range(args.epochs):
            time_ep = time.time()
            train_res = simp_utils.train_transformer_epoch(
                trainloader, 
                simplex_model, 
                criterion,
                optimizer,
                args.n_sample,
                vol_reg=1e-4,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            )

            start_ep = (epoch == 0)
            eval_ep = epoch % args.eval_freq == args.eval_freq - 1
            end_ep = epoch == args.epochs - 1
            # test_res = {'loss': None, 'accuracy': None}
            if eval_ep:
                test_res = simp_utils.eval(testloader, simplex_model, criterion)
            else:
                test_res = {'loss': None, 'accuracy': None}

            time_ep = time.time() - time_ep

            lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            values = [vv, epoch + 1, lr, 
                      train_res['loss'], train_res['accuracy'], 
                      test_res['loss'], test_res['accuracy'], time_ep]

            table = tabulate.tabulate([values], columns, 
                                      tablefmt='simple', floatfmt='8.4f')
            if epoch % 40 == 0:
                table = table.split('\n')
                table = '\n'.join([table[1]] + table)
            else:
                table = table.split('\n')[2]
            print(table, flush=True)

        checkpoint = simplex_model.state_dict()
        fname = "lr_"+str(args.lr_init)+"simplex_vertex" + str(vv) + ".pt"
        torch.save(checkpoint, savedir + fname) 
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="cifar10 simplex")

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size (default: 50)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        metavar="N",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        metavar="N",
    )


    parser.add_argument(
        "--lr_init",
        type=float,
        default=3e-2,
        metavar="LR",
        help="initial learning rate (default: 0.03)",
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=0.,
        metavar="weight_decay",
        help="weight decay",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="verts",
        help="number of vertices in simplex",
    )
    parser.add_argument(
        "--n_verts",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 100)",
    )
    
    
    parser.add_argument(
        "--n_sample",
        type=int,
        default=5,
        metavar="N",
        help="number of samples to use per iteration",
    )

    parser.add_argument(
        "--base_idx",
        type=int,
        default=0,
        metavar="N",
        help="index of base model to use",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=5,
        metavar="N",
        help="evaluate every n epochs",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar100",
        help="dataset [cifar10 or cifar100]",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=15,
        help="number of gradient accumulation steps",
    )
    args = parser.parse_args()

    main(args)
