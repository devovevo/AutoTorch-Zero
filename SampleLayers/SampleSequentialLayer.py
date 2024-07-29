import torch
from tqdm import tqdm

from BaseLayer import Base

from SampleLayer import Sample

class SampleSequential(Base, Sample):
    def __init__(self, dim, max_depth, layers, grid=None, lr=0.001, loss_fn = torch.nn.MSELoss()):
        Base.__init__(self, dim)
        
        self.max_depth = max_depth
        self.path_length = max_depth
        
        self.layers = layers

        self.grid = torch.nn.ModuleList([ torch.nn.ModuleList([layer(dim) for layer in layers]) for _ in range(max_depth) ]) if grid is None else grid

        Sample.__init__(self, lr, loss_fn)

    def forward(self, path, x):
        for i, layer in enumerate(path):
            x = self.grid[i][layer](x)

        return x
    
    def flops(self, path):
        return sum(self.grid[i][layer].flops() for i, layer in enumerate(path))