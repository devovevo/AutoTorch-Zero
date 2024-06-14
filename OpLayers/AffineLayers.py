import torch
from OpLayers.OpLayer import OpLayer

class MatMul(OpLayer):
  def __init__(self, dim):
    super().__init__()

    self.A = torch.nn.Parameter(torch.randn(dim, dim))

  def forward(self, x):
    return torch.matmul(self.A, x)
  
  def resize(self, new_dim):
    self.dim = new_dim
    new_A = torch.nn.Parameter(torch.empty(new_dim, new_dim))

    overlap = min(self.dim, new_dim)
    new_A.data[:overlap, :overlap] = self.A.data[:overlap, :overlap]
    new_A.data[:overlap, overlap:] = torch.randn(overlap, new_dim - overlap)

    self.A = new_A

  def flops(self):
    return 2 * self.dim ** 2
  
class AddBias(OpLayer):
  def __init__(self, dim):
    super().__init__()

    self.b = torch.nn.Parameter(torch.randn(dim))

  def forward(self, x):
    return x + self.b
  
  def resize(self, new_dim):
    self.dim = new_dim
    new_b = torch.nn.Parameter(torch.empty(new_dim))

    overlap = min(self.dim, new_dim)
    new_b.data[:overlap] = self.b.data[:overlap]
    new_b.data[overlap:] = torch.randn(new_dim - overlap)

    self.b = new_b

  def flops(self):
    return self.dim