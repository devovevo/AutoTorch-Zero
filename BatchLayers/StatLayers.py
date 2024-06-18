import torch

from BaseLayer import BaseLayer
    
class Mean(BaseLayer):
    def __init__(self, dim):
        super().__init__(dim)

    def forward(self, x):
        return torch.mean(x, dim=0)

    def flops(self):
        return self.dim
    
    def num_layers(self):
        return 1
    
class Variance(BaseLayer):
    def __init__(self, dim):
        super().__init__(dim)

    def forward(self, x):
        return torch.var(x, dim=0)

    def flops(self):
        return self.dim
    
    def num_layers(self):
        return 1