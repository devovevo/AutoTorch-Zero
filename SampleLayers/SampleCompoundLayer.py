import torch

from BaseLayer import Base

from SampleLayer import Sample

class SampleCompound(Base, Sample):
    def __init__(self, dim, max_dim, depth, layers, compound_layers, grid=None, lr=0.001, loss_fn = torch.nn.MSELoss()):
        Base.__init__(self, dim)
        
        self.max_dim = max_dim

        self.max_depth = depth
        self.path_length = depth + 1
        
        self.layers = layers

        self.grid = torch.nn.ModuleList([ torch.nn.ModuleList([layer(max_dim) for layer in layers]) for _ in range(depth) ]) if grid is None else grid
        self.compound_layers = torch.nn.ModuleList([ layer(dim, depth, layers, self.grid) for layer in compound_layers ])

        Sample.__init__(self, lr, loss_fn)

    def forward(self, path, x):
        return self.compound_layers[path[0]].forward(path[1:], x)
    
    def flops(self, path):
        return self.compound_layers[path[0]].flops(path[1:])