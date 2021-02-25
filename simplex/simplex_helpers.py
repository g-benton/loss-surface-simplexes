import math
import torch
import gpytorch
import utils
import time

def volume_loss(model):
    cdist = gpytorch.kernels.Kernel().covar_dist
    n_vert = model.n_vert
        
    mat = torch.ones(n_vert+1, n_vert+1) - torch.eye(n_vert + 1)
    
    ## compute distance between parameters ##
    n_par = int(sum([p.numel() for p in model.parameters()])/n_vert)
    par_vecs = torch.zeros(n_vert, n_par)
    for ii in range(n_vert):
        par_vecs[ii, :] = utils.flatten([p for p in model.net.parameters()][ii::n_vert])
    
    dist_mat = cdist(par_vecs, par_vecs).pow(2)
    mat[:n_vert, :n_vert] = dist_mat
    
    norm = (math.factorial(n_vert-1)**2) * (2. ** (n_vert-1))
    return torch.abs(torch.det(mat)).div(norm)


def complex_volume(model, ind):
    cdist = gpytorch.kernels.Kernel().covar_dist
    n_vert = len(model.simplicial_complex[ind])
    total_vert = model.n_vert
        
    mat = torch.ones(n_vert+1, n_vert+1) - torch.eye(n_vert + 1)
    
    ## compute distance between parameters ##
    temp_pars = [p for p in model.net.parameters()][0::total_vert]
    n_par = int(sum([p.numel() for p in temp_pars]))
    par_vecs = torch.zeros(n_vert, n_par).to(temp_pars[0].device)
    for ii, vv in enumerate(model.simplicial_complex[ind]):
        par_vecs[ii, :] = utils.flatten([p for p in model.net.parameters()][vv::total_vert])

    dist_mat = cdist(par_vecs, par_vecs).pow(2)
    mat[:n_vert, :n_vert] = dist_mat

    norm = (math.factorial(n_vert-1)**2) * (2. ** (n_vert-1))
    return torch.abs(torch.det(mat)).div(norm)