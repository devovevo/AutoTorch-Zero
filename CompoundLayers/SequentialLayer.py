import torch

from BaseLayer import Base

from OpLayers.ScalarFuncLayers import Identity

class Sequential(Base):
    def __init__(self, dim, layers=[]):
        Base.__init__(self, dim)

        if layers == []:
            layers = [Identity(dim)]

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def flops(self):
        return sum(layer.flops() for layer in self.layers)
    
    def num_layers(self):
        return sum(layer.num_layers() for layer in self.layers)
    
    def copy(self, weights=False):
        return Sequential(self.dim, [layer.copy(weights) for layer in self.layers])