import torch
from OpLayers.OpLayer import OpLayer

class Exp(OpLayer):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)
    
    def resize(self, new_dim):
        self.dim = new_dim
    
    def flops(self):
        return self.in_features

class Reciprocal(OpLayer):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x):
        return torch.reciprocal(x)
    
    def resize(self, new_dim):
        self.dim = new_dim
    
    def flops(self):
        return self.in_features