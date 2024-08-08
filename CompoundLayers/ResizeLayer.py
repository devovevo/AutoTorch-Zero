import torch

from BaseLayer import Base

from CompoundLayers.SequentialLayer import Sequential

from OpLayers.AffineLayers import LinearResize

class Resize(Base):
    def __init__(self, in_dim, latent_dim, out_dim):
        Base.__init__(self, in_dim)

        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim

    def __init__(self, in_dim, latent_dim, out_dim, latent_layers=[], first=None, last=None):
        Base.__init__(self, in_dim)
        
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.out_dim = out_dim
        
        if first is None:
            self.first = LinearResize(in_dim, latent_dim, first)
        else:
            self.first = first

        self.latent = Sequential(latent_dim, latent_layers)

        if last is None:
            self.last = LinearResize(latent_dim, out_dim, last)
        else:
            self.last = last

    def forward(self, x):
        x = self.first(x)
        x = self.latent(x)
        x = self.last(x)
        
        return x
    
    def flops(self):
        return 2 * self.in_dim * self.latent_dim + self.latent.flops() + 2 * self.latent_dim * self.out_dim
    
    def num_layers(self):
        return 1 + self.latent.num_layers()
    
    def copy(self, weights=False):
        return self.__class__(self.in_dim, self.latent_dim, self.out_dim, self.latent.copy(weights).layers, self.first.copy(weights), self.last.copy(weights))
    
class ResizeAmount(Resize):
    def copy(self, weights=False):
        return self.__class__(self.in_dim, latent_layers=[], first=self.first.copy(weights), last=self.last.copy(weights))

class ResizeFive(ResizeAmount):
    def __init__(self, in_dim, latent_layers=[], first=None, last=None):
        ResizeAmount.__init__(self, in_dim, in_dim * 5, in_dim, latent_layers, first, last)

class ResizeFifth(ResizeAmount):
    def __init__(self, in_dim, latent_layers=[], first=None, last=None):
        ResizeAmount.__init__(self, in_dim, max(in_dim // 5, 1), in_dim, latent_layers, first, last)

class ResizeTwo(ResizeAmount):
    def __init__(self, in_dim, latent_layers=[], first=None, last=None):
        ResizeAmount.__init__(self, in_dim, in_dim * 2, in_dim, latent_layers, first, last)

class ResizeHalf(ResizeAmount):
    def __init__(self, in_dim, latent_layers=[], first=None, last=None):
        ResizeAmount.__init__(self, in_dim, max(in_dim // 2, 1), in_dim, latent_layers, first, last)