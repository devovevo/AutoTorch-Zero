import torch

from MutableLayer import Mutable
from MutableLayers.MutableSequentialLayer import MutableSequential

from CombineLayers.ElementWiseLayers import Add, Multiply

from CompoundLayers.SplitLayer import Split

class MutableSplit(Split, Mutable):
    def __init__(self, dim, combine, left=[], right=[]):
        super().__init__(dim, combine)

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

    def copy(self):
        return MutableSplit(self.dim, self.left.copy(), self.right.copy(), self.combine.copy())
    
class MutableAddSplit(MutableSplit):
    def __init__(self, dim, left=[], right=[]):
        super().__init__(dim, Add(dim), left, right)
    
    def copy(self):
        return MutableAddSplit(self.dim, [layer.copy() for layer in self.left.layers], [layer.copy() for layer in self.right.layers])

class MutableMulSplit(MutableSplit):
    def __init__(self, dim, left=[], right=[]):
        super().__init__(dim, Multiply(dim), left, right)
    
    def copy(self):
        return MutableMulSplit(self.dim, [layer.copy() for layer in self.left.layers], [layer.copy() for layer in self.right.layers])