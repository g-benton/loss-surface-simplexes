import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair
from scipy.special import binom
import sys
sys.path.append("..")
import utils
from simplex_helpers import complex_volume

class SimplicialComplex(Module):
    def __init__(self, n_simplex):
        super(SimplicialComplex, self).__init__()
        self.n_simplex = n_simplex
    
    def forward(self, complex_modex):
        
        ## first need to pick a simplex to sample from ##
        vols = []
        n_verts = []
        for ii in range(self.n_simplex):
            vols.append(complex_volume(complex_model, ii))
            n_verts.append(len(complex_model.simplexes[ii]))

        norm = sum(vols)
        vol_cumsum = np.cumsum([vv/norm for vv in vols])
        simp_ind = np.min(np.where(np.random.rand(1) < vol_cumsum)[0])
        
        ## sample weights for simplex
        exps = [-(torch.rand(1)).log().item() for _ in range(n_verts[simp_ind])]
        total = sum(exps)
        exps = [exp/total for exp in exps]
        
        ## now assign vertex weights out
        vert_weights = [0] * complex_model.n_vert
        for ii, vert in enumerate(complex_model.simplexes[simp_ind]):
            vert_weights[vert] = exps[ii]

        return vert_weights
        
        

class Simplex(Module):
    def __init__(self, n_vert):
        super(Simplex, self).__init__()
        self.n_vert = n_vert
        self.register_buffer('range', torch.arange(0, float(n_vert)))

    def forward(self, t):
        exps = [-torch.log(torch.rand(1)).item() for _ in range(self.n_vert)]
        total = sum(exps)

        return [exp/total for exp in exps]

class PolyChain(Module):
    def __init__(self, num_bends):
        super(PolyChain, self).__init__()
        self.num_bends = num_bends
        self.register_buffer('range', torch.arange(0, float(num_bends)))

    def forward(self, t):
        t_n = t * (self.num_bends - 1)
        return torch.max(self.range.new([0.0]), 1.0 - torch.abs(t_n - self.range))


class SimplexModule(Module):

    def __init__(self, fix_points, parameter_names=()):
        super(SimplexModule, self).__init__()
        self.fix_points = fix_points
        self.num_bends = len(self.fix_points)
        self.parameter_names = parameter_names
        self.l2 = 0.0

    def compute_weights_t(self, coeffs_t):
        w_t = [None] * len(self.parameter_names)
        self.l2 = 0.0
        for i, parameter_name in enumerate(self.parameter_names):
            for j, coeff in enumerate(coeffs_t):
                parameter = getattr(self, '%s_%d' % (parameter_name, j))
                if parameter is not None:
                    if w_t[i] is None:
                        w_t[i] = parameter * coeff
                    else:
                        w_t[i] += parameter * coeff
            if w_t[i] is not None:
                self.l2 += torch.sum(w_t[i] ** 2)
        return w_t


class Linear(SimplexModule):

    def __init__(self, in_features, out_features, fix_points, bias=True):
        super(Linear, self).__init__(fix_points, ('weight', 'bias'))
        self.in_features = in_features
        self.out_features = out_features

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter(torch.Tensor(out_features, in_features), requires_grad=not fixed)
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(out_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.linear(input, weight_t, bias_t)


class Conv2d(SimplexModule):

    def __init__(self, in_channels, out_channels, kernel_size, fix_points, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(fix_points, ('weight', 'bias'))
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        for i, fixed in enumerate(self.fix_points):
            self.register_parameter(
                'weight_%d' % i,
                Parameter(
                    torch.Tensor(out_channels, in_channels // groups, *kernel_size),
                    requires_grad=not fixed
                )
            )
        for i, fixed in enumerate(self.fix_points):
            if bias:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(out_channels), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        for i in range(self.num_bends):
            getattr(self, 'weight_%d' % i).data.uniform_(-stdv, stdv)
            bias = getattr(self, 'bias_%d' % i)
            if bias is not None:
                bias.data.uniform_(-stdv, stdv)

    def forward(self, input, coeffs_t):
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.conv2d(input, weight_t, bias_t, self.stride,
                        self.padding, self.dilation, self.groups)


class _BatchNorm(SimplexModule):
    _version = 2

    def __init__(self, num_features, fix_points, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(fix_points, ('weight', 'bias'))
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.l2 = 0.0
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'weight_%d' % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('weight_%d' % i, None)
        for i, fixed in enumerate(self.fix_points):
            if self.affine:
                self.register_parameter(
                    'bias_%d' % i,
                    Parameter(torch.Tensor(num_features), requires_grad=not fixed)
                )
            else:
                self.register_parameter('bias_%d' % i, None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            for i in range(self.num_bends):
                getattr(self, 'weight_%d' % i).data.uniform_()
                getattr(self, 'bias_%d' % i).data.zero_()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input, coeffs_t):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum
        weight_t, bias_t = self.compute_weights_t(coeffs_t)
        return F.batch_norm(
            input, self.running_mean, self.running_var, weight_t, bias_t,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class BatchNorm2d(_BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))


class SimplexNet(Module):
    def __init__(self, n_output, architecture, n_vert, fix_points=None,
                 architecture_kwargs={}, simplicial_complex=None):
        super(SimplexNet, self).__init__()
        self.n_output = n_output
        self.n_vert = n_vert
        if fix_points is not None:
            self.fix_points = fix_points
        else:
            self.fix_points = n_vert * [False]
            
        if simplicial_complex is None:
            simplicial_complex = {0:[ii for ii in range(n_vert)]}
            
        self.simplicial_complex = simplicial_complex
        self.n_simplex = len(simplicial_complex)
        self.architecture = architecture
        self.architecture_kwargs = architecture_kwargs
        self.net = self.architecture(n_output, fix_points=self.fix_points, **architecture_kwargs)
        self.simplex_modules = []
        for module in self.net.modules():
            if issubclass(module.__class__, SimplexModule):
                self.simplex_modules.append(module)

    def import_base_parameters(self, base_model, index):
        parameters = list(self.net.parameters())[index::self.n_vert]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            parameter.data.copy_(base_parameter.data)

    def import_base_buffers(self, base_model):
        for buffer, base_buffer in zip(self.net.buffers(), base_model.buffers()):
            buffer.data.copy_(base_buffer.data)

    def export_base_parameters(self, base_model, index):
        parameters = list(self.net.parameters())[index::self.n_vert]
        base_parameters = base_model.parameters()
        for parameter, base_parameter in zip(parameters, base_parameters):
            base_parameter.data.copy_(parameter.data)

    def init_linear(self):
        parameters = list(self.net.parameters())
        for i in range(0, len(parameters), self.num_bends):
            weights = parameters[i:i+self.num_bends]
            for j in range(1, self.num_bends - 1):
                alpha = j * 1.0 / (self.num_bends - 1)
                weights[j].data.copy_(alpha * weights[-1].data + (1.0 - alpha) * weights[0].data)

    def weights(self, t):
        coeffs_t = self.vertex_weights()
        weights = []
        for module in self.simplex_modules:
            weights.extend([w for w in module.compute_weights_t(coeffs_t) if w is not None])
        return np.concatenate([w.detach().cpu().numpy().ravel() for w in weights])

    def forward(self, input, t=None):
        if t is None:
            t = input.data.new(1).uniform_()
        coeffs_t = self.vertex_weights()
        output = self.net(input, coeffs_t)
        return output
    
    def compute_center_weights(self):
        temp = [p for p in self.net.parameters()][0::self.n_vert]
        n_par = sum([p.numel() for p in temp])
        ## assign mean of old pars to new vertex ##
        par_vecs = self.par_vectors()
        
        return par_vecs.mean(0).unsqueeze(0)
    
    def par_vectors(self):
        temp = [p for p in self.net.parameters()][0::self.n_vert]
        n_par = sum([p.numel() for p in temp])
        ## assign mean of old pars to new vertex ##
        par_vecs = torch.zeros(self.n_vert, n_par).to(temp[0].device)
        
        for ii in range(self.n_vert):
            temp = [p for p in self.net.parameters()][ii::self.n_vert]
            par_vecs[ii, :] = utils.flatten(temp)
    
        return par_vecs
    
    def add_vert(self, to_simplexes=[0]):
        
        self.fix_points = [True] * self.n_vert + [False]
        new_model = self.architecture(self.n_output, 
                                      fix_points=self.fix_points,
                                      **self.architecture_kwargs)
        
        ## assign osld pars to new model ##
        for index in range(self.n_vert):
            old_parameters = list(self.net. parameters())[index::self.n_vert]
            new_parameters = list(new_model.parameters())[index::(self.n_vert+1)]
            for old_par, new_par in zip(old_parameters, new_parameters):
                new_par.data.copy_(old_par.data)
        
        new_parameters = list(new_model.parameters())
        new_parameters = new_parameters[(self.n_vert)::(self.n_vert+1)]
        n_par = sum([p.numel() for p in new_parameters])
        ## assign mean of old pars to new vertex ##
        par_vecs = torch.zeros(self.n_vert, n_par).to(new_parameters[0].device)
        for ii in range(self.n_vert):
            temp = [p for p in self.net.parameters()][ii::self.n_vert]
            par_vecs[ii, :] = utils.flatten(temp)

        center_pars = torch.mean(par_vecs,  0).unsqueeze(0)
        center_pars = utils.unflatten_like(center_pars, new_parameters)
        for cntr, par in zip(center_pars, new_parameters):
            par.data = cntr.to(par.device)
        
        ## update self values ##
        self.n_vert += 1
        self.net = new_model
        self.simplex_modules = []
        for module in self.net.modules():
            if issubclass(module.__class__, SimplexModule):
                self.simplex_modules.append(module)
        
        for cc in to_simplexes:
            self.simplicial_complex[cc].append(self.n_vert-1)
        
        return
    
    
    def vertex_weights(self):

        ## first need to pick a simplex to sample from ##
        simp_ind = np.random.randint(self.n_simplex)
        vols = []
        n_verts = []
        for ii in range(self.n_simplex):
#             vols.append(complex_volume(self, ii))
            n_verts.append(len(self.simplicial_complex[ii]))

        ## sample weights for simplex
        exps = [-(torch.rand(1)).log().item() for _ in range(n_verts[simp_ind])]
        total = sum(exps)
        exps = [exp/total for exp in exps]

        ## now assign vertex weights out
        vert_weights = [0] * self.n_vert
        for ii, vert in enumerate(self.simplicial_complex[simp_ind]):
            vert_weights[vert] = exps[ii]

        return vert_weights
    
    
    def total_volume(self, vol_function=complex_volume):
        vol = 0
#         for simp in range(self.n_simplex):
#             vol += complex_volume(self, simp)
        vol = complex_volume(self, 0)
        return vol