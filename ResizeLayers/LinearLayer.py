import torch
from ResizeLayers.ResizeLayer import ResizeLayer

class LinearLayer(ResizeLayer):
    def __init__(self, prev_dim, new_dim):
        super().__init__(prev_dim, new_dim)

        self.A = torch.nn.Parameter(torch.randn(new_dim, prev_dim))

    def forward(self, x):
        return torch.matmul(self.A, x)
    
    def resize(self, new_dim):
        self.new_dim = new_dim
        new_A = torch.nn.Parameter(torch.empty(new_dim, self.prev_dim))

        overlap = min(self.new_dim, self.prev_dim)
        new_A.data[:overlap, :overlap] = self.A.data[:overlap, :overlap]
        new_A.data[overlap:, :overlap] = torch.randn(new_dim - overlap, overlap)

        self.A = new_A
    
    def flops(self):
        return 2 * self.new_dim * self.prev_dim