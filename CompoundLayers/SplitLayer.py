import torch

from BaseLayer import Base

from CompoundLayers.SequentialLayer import Sequential

from CombineLayers.ElementWiseLayers import Add, Multiply, Maximum, Power

from MutableLayers.MutableSequentialLayer import MutableSequential

from OpLayers.ScalarFuncLayers import Identity

class Split(Base):
    def __init__(self, dim, combine):
        Base.__init__(self, dim)

        self.combine = combine

    def __init__(self, dim, combine, left=[], right=[]):
        Base.__init__(self, dim)

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

        return num_layers
    
    def copy(self, weights=False):
        return self.__class__(self.dim, self.combine.copy(weights), self.left.copy(weights).layers, self.right.copy(weights).layers)

class FuncSplit(Split):
    def copy(self, weights=False):
        return self.__class__(self.dim, self.left.copy(weights).layers, self.right.copy(weights).layers)

class AddSplit(FuncSplit):
    def __init__(self, dim, left=[], right=[]):
        FuncSplit.__init__(self, dim, Add(dim), left, right)

class MulSplit(FuncSplit):
    def __init__(self, dim, left=[], right=[]):
        FuncSplit.__init__(self, dim, Multiply(dim), left, right)
    
class MaxSplit(FuncSplit):
    def __init__(self, dim, left=[], right=[]):
        FuncSplit.__init__(self, dim, Maximum(dim), left, right)
    
class PowSplit(FuncSplit):
    def __init__(self, dim, left=[], right=[]):
        FuncSplit.__init__(self, dim, Power(dim), left, right)