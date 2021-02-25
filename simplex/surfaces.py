import torch
import torch.nn as nn
import numpy as np
import math
import utils


def get_basis(model, anchor=0, base1=1, base2=2):
    n_vert = model.n_vert
    n_par = int(sum([p.numel() for p in model.parameters()])/n_vert)
    
    if n_vert <= 2:
        return torch.randn(n_par, 1), torch.randn(n_par, 1)
    else:
        par_vecs = torch.zeros(n_vert, n_par)
        if torch.has_cuda:
            par_vecs = par_vecs.cuda()
        for ii in range(n_vert):
            temp_pars = [p for p in model.net.parameters()][ii::n_vert]
            par_vecs[ii, :] = utils.flatten(temp_pars)
            
        
        first_pars = torch.cat((n_vert * [par_vecs[anchor, :].unsqueeze(0)]))
        diffs = (par_vecs - first_pars)
        dir1 = diffs[base1, :]
        dir2 = diffs[base2, :]
        
        ## now gram schmidt these guys ##
        vu = dir2.squeeze().dot(dir1.squeeze())
        uu = dir1.squeeze().dot(dir1.squeeze())

        dir2 = dir2 - dir1.mul(vu).div(uu)

        ## normalize ##
        dir1 = dir1.div(dir1.norm())
        dir2 = dir2.div(dir2.norm())

        return dir1.unsqueeze(-1), dir2.unsqueeze(-1)

def compute_loss_surface(model, train_x, train_y, v1, v2,
                        loss, n_pts=50, range_=10.):
    
    start_pars = model.state_dict()
    vec_len = torch.linspace(-range_.item(), range_.item(), n_pts)
    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts)
    with torch.no_grad():
        ## loop and get loss at each point ##
        for ii in range(n_pts):
            for jj in range(n_pts):
                perturb = v1.mul(vec_len[ii]) + v2.mul(vec_len[jj])
                # print(perturb.shape)
                perturb = utils.unflatten_like(perturb.t(), model.parameters())
                for i, par in enumerate(model.parameters()):
                    par.data = par.data + perturb[i].to(par.device)

                loss_surf[ii, jj] = loss(model(train_x), train_y)

                model.load_state_dict(start_pars)

    X, Y = np.meshgrid(vec_len, vec_len)
    return X, Y, loss_surf


def compute_loader_loss(model, loader, loss, n_batch,
                       device=torch.device("cuda:0")):
    total_loss = torch.tensor([0.])
    for i, data in enumerate(loader):
        if i < n_batch:
            x, y = data
            x, y = x.to(device), y.to(device)

            preds = model(x)
            total_loss += loss(preds, y).item()
        else:
            break

    return total_loss

def compute_loss_surface_loader(model, loader, v1, v2,
                                loss=torch.nn.CrossEntropyLoss(),
                                n_batch=10, n_pts=50, range_=10.,
                               device=torch.device("cuda:0")):
    
    start_pars = model.state_dict()
    vec_len = torch.linspace(-range_.item(), range_.item(), n_pts)
    ## init loss surface and the vector multipliers ##
    loss_surf = torch.zeros(n_pts, n_pts)
    with torch.no_grad():
        ## loop and get loss at each point ##
        for ii in range(n_pts):
            for jj in range(n_pts):
                perturb = v1.mul(vec_len[ii]) + v2.mul(vec_len[jj])
                # print(perturb.shape)
                perturb = utils.unflatten_like(perturb.t(), model.parameters())
                for i, par in enumerate(model.parameters()):
                    par.data = par.data + perturb[i].to(par.device)
                    
                loss_surf[ii, jj] = compute_loader_loss(model, loader,
                                                        loss, n_batch,
                                                        device=device)

                model.load_state_dict(start_pars)

    X, Y = np.meshgrid(vec_len, vec_len)
    return X, Y, loss_surf
