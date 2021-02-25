import torch
import torch.nn as nn
from simplex_models import Linear as SimplexLinear

class BasicNet(nn.Module):
    """docstring for SimpleNet."""
    def __init__(self, out_dim, in_dim=2, hidden_size=10,
                activation=torch.nn.ReLU(), bias=True):
        super(BasicNet, self).__init__()

        self.activation = activation
        self.fc1 = nn.Linear(in_dim, hidden_size, bias=bias)
        self.fc2 = nn.Linear(hidden_size, 
                      hidden_size, bias=bias)
        self.fc7 = nn.Linear(hidden_size, 
                      hidden_size, bias=bias)
        self.fc3 = nn.Linear(hidden_size, 
                      hidden_size, bias=bias)
        self.fc4 = nn.Linear(hidden_size, 
                      hidden_size, bias=bias)
        self.fc5 = nn.Linear(hidden_size, 
                      hidden_size, bias=bias)
        self.fc6 = nn.Linear(hidden_size, 
                      hidden_size, bias=bias)

        self.fc7 = nn.Linear(hidden_size, 
                      hidden_size, bias=bias)
        self.fc8 = nn.Linear(hidden_size, 
                      out_dim, bias=bias)


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)        
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.fc5(x)
        x = self.activation(x)
        x = self.fc6(x)
        x = self.activation(x)
        x = self.fc7(x)
        x = self.activation(x)
        x = self.fc8(x)
        
        return x

    
class BasicSimplex(nn.Module):
    """docstring for SimpleNet."""
    def __init__(self, out_dim, fix_points, in_dim=2, hidden_size=10,
                activation=torch.nn.ReLU(), bias=True):
        super(BasicSimplex, self).__init__()

        ## initialize the network ##
        self.activation = activation
        self.fc1 = SimplexLinear(in_dim, hidden_size, bias=bias,
                                 fix_points=fix_points)
        self.fc2 = SimplexLinear(hidden_size, 
                      hidden_size, bias=bias,
                      fix_points=fix_points)
        self.fc7 = SimplexLinear(hidden_size, 
                      hidden_size, bias=bias,
                      fix_points=fix_points)
        self.fc3 = SimplexLinear(hidden_size, 
                      hidden_size, bias=bias,
                      fix_points=fix_points)
        self.fc4 = SimplexLinear(hidden_size, 
                      hidden_size, bias=bias,
                      fix_points=fix_points)
        self.fc5 = SimplexLinear(hidden_size, 
                      hidden_size, bias=bias,
                      fix_points=fix_points)
        self.fc6 = SimplexLinear(hidden_size, 
                      hidden_size, bias=bias,
                      fix_points=fix_points)
        self.fc7 = SimplexLinear(hidden_size, 
                      hidden_size, bias=bias,
                      fix_points=fix_points)
        self.fc8 = SimplexLinear(hidden_size, 
                      out_dim, bias=bias,
                      fix_points=fix_points)


    def forward(self, x, coeffs_t):
        x = self.fc1(x, coeffs_t)
        x = self.activation(x)
        x = self.fc2(x, coeffs_t)
        x = self.activation(x)        
        x = self.fc3(x, coeffs_t)
        x = self.activation(x)
        x = self.fc4(x, coeffs_t)
        x = self.activation(x)
        x = self.fc5(x, coeffs_t)
        x = self.activation(x)
        x = self.fc6(x, coeffs_t)
        x = self.activation(x)
        x = self.fc7(x, coeffs_t)
        x = self.activation(x)
        x = self.fc8(x, coeffs_t)
        
        return x
