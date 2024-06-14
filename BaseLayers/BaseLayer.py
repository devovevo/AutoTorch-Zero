import torch

class BaseLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError
    
    def flops(self):
        raise NotImplementedError