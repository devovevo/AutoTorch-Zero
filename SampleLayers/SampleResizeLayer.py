import torch
from tqdm import tqdm

from BaseLayer import Base

from OpLayers.AffineLayers import LinearResize

from SampleLayer import Sample
from SampleLayers.SampleSequentialLayer import SampleSequential

class SampleResize(Base, Sample):
    def __init__(self, in_dim, latent_dim, out_dim, max_depth, layers, grid=None, lr=0.001, loss_fn = torch.nn.MSELoss()):
        Base.__init__(self, in_dim)
        
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.max_depth = max_depth
        self.path_length = max_depth
        
        self.layers = layers

        self.first = LinearResize(in_dim, latent_dim)
        self.latent = SampleSequential(latent_dim, max_depth, layers, grid)
        self.last = LinearResize(latent_dim, out_dim)

        Sample.__init__(self, lr, loss_fn)

    def forward(self, path, x):
        x = self.first(x)
        x = self.latent.forward(path, x)
        
        return self.last(x)

    def flops(self, path):
        return self.first.flops() + self.latent.flops(path) + self.last.flops()
    
class SampleResizeFive(SampleResize):
    def __init__(self, in_dim, max_depth, layers, grid=None, lr=0.001, loss_fn = torch.nn.MSELoss(),):
        super().__init__(in_dim, in_dim * 5, in_dim, max_depth, layers, grid, lr, loss_fn)

class SampleResizeFifth(SampleResize):
    def __init__(self, in_dim, max_depth, layers, grid=None, lr=0.001, loss_fn = torch.nn.MSELoss()):
        super().__init__(in_dim, max(in_dim // 5, 1), in_dim, max_depth, layers, grid, lr, loss_fn)