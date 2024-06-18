import torch

from BaseLayer import BaseLayer

class Pow(BaseLayer):
    def __init__(self, dim):
        super().__init__(dim)
        
        self.power = torch.nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return torch.pow(x, self.power)
    
    def flops(self):
        return self.dim
    
class Scale(BaseLayer):
    def __init__(self, dim):
        super().__init__(dim)
        
        self.scale = torch.nn.Parameter(torch.randn(dim))

    def forward(self, x):
        return x * self.scale
    
    def flops(self):
        return self.dim