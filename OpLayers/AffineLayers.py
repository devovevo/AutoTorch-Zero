import torch

from SimpleLayer import Simple

class LinearResize(Simple):
  def __init__(self, in_dim, out_dim, A=None):
    Simple.__init__(self, in_dim)

    self.in_dim = in_dim
    self.out_dim = out_dim

    if A is None:
      self.A = torch.nn.Parameter(torch.randn(in_dim, out_dim))
    else:
      self.A = torch.nn.Parameter(A)

  def forward(self, x):
    return torch.matmul(x, self.A[:x.shape[1], :])
  
  def flops(self):
    return self.in_dim * self.out_dim
  
  def copy(self, weights=False):
    if weights:
      return LinearResize(self.in_dim, self.out_dim, self.A.clone().detach().requires_grad_(True))
    else:
      return LinearResize(self.in_dim, self.out_dim)
  
  def extra_repr(self):
    return f"in_dim={self.in_dim}, out_dim={self.out_dim}, A={self.A.detach().numpy()}"

class Linear(LinearResize):
  def __init__(self, dim, A=None):
    LinearResize.__init__(self, dim, dim, A)

  def forward(self, x):
    return torch.matmul(x, self.A[:x.shape[1], :x.shape[1]])

  def copy(self, weights=False):
    if weights:
      return Linear(self.dim, self.A.clone().detach().requires_grad_(True))
    else:
      return Linear(self.dim)
  
class AddBias(Simple):
  def __init__(self, dim, b=None):
    Simple.__init__(self, dim)

    if b is None:
      self.b = torch.nn.Parameter(torch.randn(dim))
    else:
      self.b = torch.nn.Parameter(b)

  def forward(self, x):
    return x + self.b[:x.shape[1]]
  
  def copy(self, weights=False):
    if weights:
      return AddBias(self.dim, self.b.clone().detach().requires_grad_(True))
    else:
      return AddBias(self.dim)
  
  def extra_repr(self):
    return f"dim={self.dim}, b={self.b.detach().numpy()}"