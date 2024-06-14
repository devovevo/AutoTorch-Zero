import torch

class BaseLayer(torch.nn.Module):
    def forward(self, x):
        raise NotImplementedError
    
    def resize(self, new_dim):
        raise NotImplementedError
    
    def flops(self):
        raise NotImplementedError