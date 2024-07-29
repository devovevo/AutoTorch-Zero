import torch
from tqdm import tqdm

from BaseLayer import Base

from CombineLayers.ElementWiseLayers import Add, Multiply, Maximum, Power

from SampleLayer import Sample
from SampleLayers.SampleSequentialLayer import SampleSequential

class SampleSplit(Base, Sample):
    def __init__(self, dim, max_depth, layers, combine, grid=None, lr=0.001, loss_fn = torch.nn.MSELoss()):
        Base.__init__(self, dim)
        
        self.max_depth = max_depth
        self.path_length = max_depth
        
        self.layers = layers

        self.left_depth = max_depth // 2
        self.right_depth = max_depth - self.left_depth

        if grid is not None:
            self.left = SampleSequential(dim, self.left_depth, layers, grid[:self.left_depth])
            self.right = SampleSequential(dim, self.right_depth, layers, grid[self.left_depth:self.max_depth])
        else:
            self.left = SampleSequential(dim, self.left_depth, layers)
            self.right = SampleSequential(dim, self.right_depth, layers)

        self.combine = combine

        Sample.__init__(self, lr, loss_fn)

    def forward(self, path, x):
        left_path = path[:self.left_depth]
        right_path = path[self.left_depth:self.max_depth]

        return self.combine(self.left.forward(left_path, x), self.right.forward(right_path, x))
    
    def flops(self, path):
        left_path = path[:self.left_depth]
        right_path = path[self.left_depth:self.max_depth]

        return self.left.flops(left_path) + self.right.flops(right_path) + self.combine.flops()
    
class SampleSplitAdd(SampleSplit):
    def __init__(self, dim, max_depth, layers, grid=None, lr=0.001, loss_fn = torch.nn.MSELoss()):
        super().__init__(dim, max_depth, layers, Add(dim), grid, lr, loss_fn)

class SampleSplitMultiply(SampleSplit):
    def __init__(self, dim, max_depth, layers, grid=None, lr=0.001, loss_fn = torch.nn.MSELoss()):
        super().__init__(dim, max_depth, layers, Multiply(dim), grid, lr, loss_fn)

class SampleSplitMaximum(SampleSplit):
    def __init__(self, dim, max_depth, layers, grid=None, lr=0.001, loss_fn = torch.nn.MSELoss()):
        super().__init__(dim, max_depth, layers, Maximum(dim), grid, lr, loss_fn)

class SampleSplitPower(SampleSplit):
    def __init__(self, dim, max_depth, layers, grid=None, lr=0.001, loss_fn = torch.nn.MSELoss()):
        super().__init__(dim, max_depth, layers, Power(dim), grid, lr, loss_fn)