import math
import torch
import gpytorch
import utils

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
    return torch.abs(torch.det(mat)).div(norm).pow(0.5)


def complex_volume(model, ind):
    print(ind)
    cdist = gpytorch.kernels.Kernel().covar_dist
    n_vert = len(model.simplicial_complex[ind])
    total_vert = model.n_vert
        
    mat = torch.ones(n_vert+1, n_vert+1) - torch.eye(n_vert + 1)
    
    ## compute distance between parameters ##
    n_par = int(sum([p.numel() for p in model.parameters()])/n_vert)
    par_vecs = torch.zeros(n_vert, n_par)
    for ii, vv in enumerate(model.simplicial_complex[ind]):
        par_vecs[ii, :] = utils.flatten([p for p in model.net.parameters()][vv::total_vert])
    
    dist_mat = cdist(par_vecs, par_vecs).pow(2)
    mat[:n_vert, :n_vert] = dist_mat
    
    norm = (math.factorial(n_vert-1)**2) * (2. ** (n_vert-1))
    return torch.abs(torch.det(mat)).div(norm).pow(0.5)