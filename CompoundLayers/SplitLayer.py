import torch

from BaseLayer import BaseLayer

from CompoundLayers.SequentialLayer import Sequential

from OpLayers.ScalarFuncLayers import Identity
from CombineLayers.ElementWiseLayers import Add

class Split(BaseLayer):
    def __init__(self, dim, left=[], right=[], combine=None):
        super().__init__(dim)

        if combine is None:
            combine = Add(dim)

        self.left = Sequential(dim, left)
        self.right = Sequential(dim, right)

        self.combine = combine

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)

        return self.combine(left, right)
    
    def flops(self):
        flops = 0

        flops += self.left.flops()
        flops += self.right.flops()

        flops += self.combine.flops()

        return flops
    
    def num_layers(self):
        num_layers = 1

        num_layers += self.left.num_layers()
        num_layers += self.right.num_layers()

        num_layers += self.combine.num_layers()

        return num_layers