import torch

from BaseLayer import Base

class Mutable(Base):
    def insert_layer(self, index, layer):
        raise NotImplementedError
    
    def remove_layer(self, index):
        raise NotImplementedError
    
    def replace_layer(self, index, layer):
        raise NotImplementedError