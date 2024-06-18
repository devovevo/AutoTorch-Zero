import torch

from BaseLayer import BaseLayer

class Add(BaseLayer):
    def __init__(self, dim):
        super().__init__(dim)

    def forward(self, x, y):
        return x + y
    
    def flops(self):
        return self.dim
    
    def num_layers(self):
        return 1
    
class Multiply(BaseLayer):
    def __init__(self, dim):
        super().__init__(dim)

    def forward(self, x, y):
        return x * y
    
    def flops(self):
        return self.dim
    
    def num_layers(self):
        return 1
    
class Maximum(BaseLayer):
    def __init__(self, dim):
        super().__init__(dim)

    def forward(self, x, y):
        return torch.maximum(x, y)
    
    def flops(self):
        return self.dim
    
    def num_layers(self):
        return 1