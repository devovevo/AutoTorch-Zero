import torch

from SimpleLayer import Simple
    
class Mean(Simple):
    def forward(self, x):
        return torch.ones_like(x) * torch.mean(x, dim=0, keepdim=True)
    
class Variance(Simple):
    def forward(self, x):
        return torch.ones_like(x) * torch.var(x, dim=0, keepdim=True)