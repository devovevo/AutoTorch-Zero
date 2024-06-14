import torch
from BaseLayers.BaseLayer import BaseLayer

class MatMul(BaseLayer):
  def __init__(self, in_features, out_features):
    super().__init__()

    self.in_features = in_features
    self.out_features = out_features

    self.A = torch.nn.Parameter(torch.randn(in_features, out_features))

  def forward(self, x):
    return torch.matmul(self.A, x)
  
  def flops(self):
    return 2 * self.in_features * self.out_features
  
class AddBias(BaseLayer):
  def __init__(self, in_features, out_features):
    super().__init__()

    assert in_features == out_features

    self.in_features = in_features
    self.out_features = out_features

    self.b = torch.nn.Parameter(torch.randn(in_features))

  def forward(self, x):
    return x + self.b
  
  def flops(self):
    return self.in_features