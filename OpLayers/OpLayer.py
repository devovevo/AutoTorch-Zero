import torch
from BaseLayer import BaseLayer

class OpLayer(BaseLayer):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
    
    def forward(self, x):
        raise NotImplementedError
    
    def resize(self, new_dim):
        raise NotImplementedError
    
    def flops(self):
        raise NotImplementedError