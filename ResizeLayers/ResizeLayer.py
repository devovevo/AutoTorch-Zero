import torch
from BaseLayer import BaseLayer

class ResizeLayer(BaseLayer):
    def __init__(self, prev_dim, new_dim):
        super().__init__()

        self.prev_dim = prev_dim
        self.new_dim = new_dim

    def forward(self, x):
        raise NotImplementedError
    
    def resize(self, new_dim):
        raise NotImplementedError
    
    def flops(self):
        raise NotImplementedError