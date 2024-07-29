import torch

from MutableLayer import Mutable
from MutableLayers.MutableSequentialLayer import MutableSequential

from CombineLayers.ElementWiseLayers import Add, Multiply, Maximum, Power

from CompoundLayers.SplitLayer import Split

class MutableSplit(Split, Mutable):
    def __init__(self, dim, combine, left=[], right=[]):
        Split.__init__(self, dim, combine)

        self.left = MutableSequential(dim, left)
        self.right = MutableSequential(dim, right)

        self.combine = combine

    def insert_layer(self, index, layer):
        if index < self.left.num_layers():
            self.left.insert_layer(index, layer)
        elif index < self.left.num_layers() + self.right.num_layers():
            self.right.insert_layer(index - self.left.num_layers(), layer)

    def remove_layer(self, index):
        if index < self.left.num_layers() and self.left.num_layers() > 1:
            self.left.remove_layer(index)
        elif index < self.left.num_layers() + self.right.num_layers() and self.right.num_layers() > 1:
            self.right.remove_layer(index - self.left.num_layers())

    def replace_layer(self, index, layer):
        if index < self.left.num_layers():
            self.left.replace_layer(index, layer)
        elif index < self.left.num_layers() + self.right.num_layers():
            self.right.replace_layer(index - self.left.num_layers(), layer)

class MutableFuncSplit(MutableSplit):
    def copy(self, weights=False):
        return self.__class__(self.dim, self.left.copy(weights).layers, self.right.copy(weights).layers)

class MutableAddSplit(MutableFuncSplit):
    def __init__(self, dim, left=[], right=[]):
        MutableFuncSplit.__init__(self, dim, Add(dim), left, right)

class MutableMulSplit(MutableFuncSplit):
    def __init__(self, dim, left=[], right=[]):
        MutableFuncSplit.__init__(self, dim, Multiply(dim), left, right)
    
class MutableMaxSplit(MutableFuncSplit):
    def __init__(self, dim, left=[], right=[]):
        MutableFuncSplit.__init__(self, dim, Maximum(dim), left, right)
    
class MutablePowSplit(MutableFuncSplit):
    def __init__(self, dim, left=[], right=[]):
        MutableFuncSplit.__init__(self, dim, Power(dim), left, right)