import torch
from ResizeLayers.ResizeLayer import ResizeLayer

class PadLayer(ResizeLayer):
    def __init__(self, prev_dim, new_dim):
        super().__init__(prev_dim, new_dim)
    
    def forward(self, x):
        y = torch.empty(self.new_dim)

        overlap = min(self.prev_dim, self.new_dim)
        y[:overlap] = x[:overlap]
        y[overlap:] = 0

        return y
    
    def resize(self, new_dim):
        self.new_dim = new_dim

    def flops(self):
        return self.new_dim