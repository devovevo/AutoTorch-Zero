import torch

from BaseLayer import Base

from OpLayers.AffineLayers import LinearResize
from OpLayers.ScalarFuncLayers import Identity

from SampleLayer import Sample
from SampleLayers.SampleCompoundLayer import SampleCompound

class SampleChain(Base, Sample):
    def __init__(self, in_dim, max_dim, out_dim, depth, layers, compound_layers, chain_length, shared=False, lr=0.001, loss_fn = torch.nn.MSELoss()):
        Base.__init__(self, in_dim)
        
        self.max_dim = max_dim
        self.out_dim = out_dim

        self.max_depth = depth
        self.path_length = chain_length * (depth + 1) + 1

        self.layers = layers
        self.compound_layers = compound_layers

        if shared:
            self.grid = torch.nn.ModuleList([ torch.nn.ModuleList([layer(max_dim) for layer in layers]) for _ in range(depth) ])
        else:
            self.grid = None

        self.chain_length = chain_length
        self.chain = torch.nn.ModuleList([ SampleCompound(in_dim, max_dim, depth, layers, compound_layers, self.grid) for _ in range(chain_length) ])

        # self.last = torch.nn.ModuleList([ LinearResize(in_dim, out_dim) for _ in layers ])
        if in_dim == out_dim:
            self.last = Identity(out_dim)
        else:
            self.last = LinearResize(in_dim, out_dim)

        Sample.__init__(self, lr, loss_fn)

    def forward(self, path, x):
        for c in torch.arange(start=0, end=self.chain_length, step=1):
            cur_path = path[c * (self.max_depth + 1):(c + 1) * (self.max_depth + 1)]
            x = self.chain[c].forward(cur_path, x)

        return self.last.forward(x)

    def flops(self, path):
        return sum(self.chain[c].flops(path[c * (self.max_depth + 1):(c + 1) * (self.max_depth + 1)]) for c in range(self.chain_length))