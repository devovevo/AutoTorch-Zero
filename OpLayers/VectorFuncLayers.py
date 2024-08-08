import torch

from SimpleLayer import Simple

class Pow(Simple):
    def __init__(self, dim, exp=None):
        Simple.__init__(self, dim)
        
        if exp is None:
            self.exp = torch.nn.Parameter(torch.randn(dim))
        else:
            self.exp = torch.nn.Parameter(exp)

    def forward(self, x):
        return torch.pow(torch.abs(x), self.exp[:x.shape[1]]) * torch.sign(x)
    
    def copy(self, weights=False):
        if weights:
            return Pow(self.dim, self.exp.clone().detach().requires_grad_(True))
        else:
            return Pow(self.dim)
    
    def extra_repr(self):
        return f"dim={self.dim}, exp={self.exp.detach().numpy()}"
    
class Scale(Simple):
    def __init__(self, dim, scale=None):
        Simple.__init__(self, dim)
        
        if scale is None:
            self.scale = torch.nn.Parameter(torch.randn(dim))
        else:
            self.scale = torch.nn.Parameter(scale)

    def forward(self, x):
        return x * self.scale[:x.shape[1]]
    
    def copy(self, weights=False):
        if weights:
            return Scale(self.dim, self.scale.clone().detach().requires_grad_(True))
        else:
            return Scale(self.dim)
    
    def extra_repr(self):
        return f"dim={self.dim}, scale={self.scale.detach().numpy()}"