import torch
import math

from gpytorch.kernels import Kernel

def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def simplex_parameters(module, params, num_vertices):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)

        for i in range(num_vertices):
            module.register_parameter(name + "_vertex_" + str(i),
                                      torch.nn.Parameter(data.clone().detach_().requires_grad_()))

        params.append((module, name))


cdist = Kernel().covar_dist

class BasicSimplex(torch.nn.Module):
    def __init__(self, base, num_vertices = 2, fixed_points = [True, False], *args, **kwargs):
        super().__init__()
        self.params = list()
        # self.base = base(*args, **kwargs)
        self.base = base
        self.base.apply(
            lambda module: simplex_parameters(
                module=module, params=self.params, num_vertices=num_vertices
            )
        )
        self.num_vertices = num_vertices
        self._fix_points(fixed_points)
        self.n_vert = num_vertices

    def _fix_points(self, fixed_points):
        for (module, name) in self.params:
            for vertex in range(self.num_vertices):
                if fixed_points[vertex]:
                    module.__getattr__(name + "_vertex_" + str(vertex)).detach_()
                    module.__getattr__(name + "_vertex_" + str(vertex)).requires_grad_(False)
                    
    def sample(self, coeffs_t):
        for (module, name) in self.params:
            new_par = 0.
            for vertex in range(self.num_vertices):
                vert = module.__getattr__(name + "_vertex_" + str(vertex))
                new_par = new_par + vert * coeffs_t[vertex]
            module.__setattr__(name, new_par)

    def vertex_weights(self):
        exps = -torch.rand(self.num_vertices).log()
        return exps / exps.sum()

    def forward(self, X, coeffs_t=None):
        if coeffs_t is None:
            coeffs_t = self.vertex_weights()

        self.sample(coeffs_t)
        return self.base(X)

    def add_vert(self):
        return self.add_vertex()

    def add_vertex(self):
        new_vertex = self.num_vertices

        for (module, name) in self.params:
            data = 0.
            for vertex in range(self.num_vertices):
                with torch.no_grad():
                    data += module.__getattr__(name + "_vertex_" + str(vertex))
            data = data / self.num_vertices

            module.register_parameter(name + "_vertex_" + str(new_vertex),
                                      torch.nn.Parameter(data.clone().detach_().requires_grad_()))
        self.num_vertices += 1

    def total_volume(self):
        n_vert = self.num_vertices


        dist_mat = 0.
        for (module, name) in self.params:
            all_vertices = [] #* self.num_vertices
            for vertex in range(self.num_vertices):
                par = module.__getattr__(name + "_vertex_" + str(vertex))
                all_vertices.append(flatten(par))
            par_vecs = torch.stack(all_vertices)
            dist_mat = dist_mat + cdist(par_vecs, par_vecs).pow(2)

        mat = torch.ones(n_vert+1, n_vert+1) - torch.eye(n_vert + 1)
        # dist_mat = cdist(par_vecs, par_vecs).pow(2)
        mat[:n_vert, :n_vert] = dist_mat

        norm = (math.factorial(n_vert-1)**2) * (2. ** (n_vert-1))
        return torch.abs(torch.det(mat)).div(norm)
    
    def par_vectors(self):
        all_vertices_list = []
        for vertex in range(self.num_vertices):
            vertex_list = []
            for (module, name) in self.params:
                val = module.__getattr__(name + "_vertex_" + str(vertex)).detach()
                vertex_list.append(val)
            all_vertices_list.append(flatten(vertex_list))
        return torch.stack(all_vertices_list)


