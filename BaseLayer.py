from torch.nn import Module

class Base(Module):
    def __init__(self, dim):
        Module.__init__(self)

        self.dim = dim

    def forward(self, x):
        raise NotImplementedError
    
    def flops(self):
        raise NotImplementedError
    
    def num_layers(self):
        raise NotImplementedError
    
    def copy(self, weights=False):
        raise NotImplementedError
    
    def extra_repr(self):
        return f"dim={self.dim}"