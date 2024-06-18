import torch

from BaseLayer import BaseLayer

class Sequential(BaseLayer):
    def __init__(self, dim, layers=[]):
        super().__init__(dim)

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def flops(self):
        return sum(layer.flops() for layer in self.layers)
    
    def num_layers(self):
        return sum(layer.num_layers() for layer in self.layers)