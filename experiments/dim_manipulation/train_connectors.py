import tabulate
import time
import torch
import argparse

from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms

import sys
sys.path.append("../simplex/")
from models.lenet import LeNet5
from models.basic_simplex import BasicSimplex
import utils as simp_utils

def main(
    epochs=20,
    n_verts=5,
    n_sample=5,
    width_pars=[6, 16, 120, 84],
    data_path="/datasets/",
    eval_freq=2,
    model="lenet",
    savedir="./",
):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    data_train = MNIST(data_path,
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
    data_test = MNIST(data_path,
                    train=False,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()]))

    trainloader = torch.utils.data.DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(data_test, batch_size=1024, num_workers=8)

    if model == "lenet":
        nets = [LeNet5(width_pars=width_pars)] * 2
    elif model == "mlp":
        nets = [MLPmnist(width_pars=width_pars)] * 2

    [net.load_state_dict(torch.load(f)) for net, f in zip(nets, net_paths)]

    criterion = torch.nn.CrossEntropyLoss()
    

    simplex_model = BasicSimplex(nets[0], num_vertices=2, fixed_points=[True, True]).to(device)
    simplex_model.set_vertex(1, nets[1].state_dict())
        ## add a new points and train ##
    for vv in range(n_verts):
        if vv > 0:
            simplex_model.add_vert()
        simplex_model = simplex_model.to(device)
        optimizer = torch.optim.Adam(simplex_model.parameters(), lr=2e-3)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=epochs)
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
                n_sample=n_sample,
                vol_reg=1e-4,
                gradient_accumulation_steps=1,
                device=device,
            )

            # start_ep = (epoch == 0)
            eval_ep = epoch % eval_freq == eval_freq - 1
            # end_ep = epoch == epochs - 1
            # test_res = {'loss': None, 'accuracy': None}
            if eval_ep:
                test_res = simp_utils.eval(testloader, simplex_model, criterion, device)
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
        fname = "simplex_vertex" + str(vv) + ".pt"
        torch.save(checkpoint, savedir + fname) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="mnist training, varying width")

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="number of epochs for training",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=2,
        help="how often to evaluate on test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/datasets/",
        help="location of dataset",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="./",
        help="location to save checkpoints to",
    )
    parser.add_argument(
        '--width_pars', 
        nargs='+', 
        help='<Required> Set flag', 
        default=[6, 16, 120, 84],
    )
    args = parser.parse_args()
    main(**vars(args))
