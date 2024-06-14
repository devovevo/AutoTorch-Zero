import torch
from BaseLayers.BaseLayer import BaseLayer

class Exp(BaseLayer):
    def __init__(self, in_features, out_features):
        super().__init__()

        assert in_features == out_features

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return torch.exp(x)
    
    def flops(self):
        return self.in_features

class Reciprocal(BaseLayer):
    def __init__(self, in_features, out_features):
        super().__init__()

        assert in_features == out_features

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return torch.reciprocal(x)
    
    def flops(self):
        return self.in_features