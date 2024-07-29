import torch

from CompoundLayers.ResizeLayer import Resize

from OpLayers.AffineLayers import LinearResize

from MutableLayer import Mutable
from MutableLayers.MutableSequentialLayer import MutableSequential

class MutableResize(Resize, Mutable):
    def __init__(self, in_dim, latent_dim, out_dim, latent_layers=[], first=None, last=None):
        Resize.__init__(self, in_dim, latent_dim, out_dim)

        if first is None:
            self.first = LinearResize(in_dim, latent_dim, first)
        else:
            self.first = first

        self.latent = MutableSequential(latent_dim, latent_layers)

        if last is None:
            self.last = LinearResize(latent_dim, out_dim, last)
        else:
            self.last = last

    def insert_layer(self, index, layer):
        self.latent.insert_layer(index, layer)

    def remove_layer(self, index):
        self.latent.remove_layer(index)

    def replace_layer(self, index, layer):
        self.latent.replace_layer(index, layer)
        
class MutableResizeAmount(MutableResize):
    def copy(self, weights=False):
        return self.__class__(self.in_dim, latent_layers=[], first=self.first.copy(weights), last=self.last.copy(weights))

class MutableResizeFive(MutableResizeAmount):
    def __init__(self, in_dim, latent_layers=[], first=None, last=None):
        MutableResizeAmount.__init__(self, in_dim, in_dim * 5, in_dim, latent_layers, first, last)
    
class MutableResizeFifth(MutableResizeAmount):
    def __init__(self, in_dim, latent_layers=[], first=None, last=None):
        MutableResizeAmount.__init__(self, in_dim, max(in_dim // 5, 1), in_dim, latent_layers, first, last)