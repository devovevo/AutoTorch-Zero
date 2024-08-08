import torch

from BaseLayer import Base

class Simple(Base):
    def flops(self):
        return self.dim

    def num_layers(self):
        return 1
    
    def copy(self, weights=False):
        return self.__class__(self.dim)