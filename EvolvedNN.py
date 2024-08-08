import torch
import numpy as np

from tqdm import tqdm

from MutableLayers.MutableResizeLayers import MutableSequential

from OpLayers.AffineLayers import LinearResize

from TrainableLayer import Trainable

class EvolvedNN(MutableSequential, Trainable):
    def __init__(self, in_dim, out_dim, layers=[], lr=0.001, loss_fn=torch.nn.MSELoss(), last=None):
        MutableSequential.__init__(self, in_dim, layers)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.last = LinearResize(in_dim, out_dim) if last is None else last

        Trainable.__init__(self, lr=lr, loss_fn=loss_fn)

    def forward(self, x):
        x = super().forward(x)

        return self.last.forward(x)
    
    mutations = ['insert', 'remove', 'replace']
    
    def mutate(self, layers, p=[0.4, 0.2, 0.4]):
        mutation = np.random.choice(EvolvedNN.mutations, p=p)
        index = torch.randint(self.num_layers(), (1,)).item()

        match mutation:
            case 'insert':
                layer = layers[torch.randint(len(layers), (1,)).item()]
                self.insert_layer(index, layer)
            case 'remove':
                self.remove_layer(index)
            case 'replace':
                layer = layers[torch.randint(len(layers), (1,)).item()]
                self.replace_layer(index, layer)

        lr_update = torch.abs(torch.randn(1) * 0.2 + 1).item()
        self.lr = self.lr * lr_update

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

    def flops(self):
        return super().flops() + self.last.flops()
    
    def num_layers(self):
        return super().num_layers() + 1

    def copy(self, weights=False):
        return EvolvedNN(self.in_dim, self.out_dim, super().copy(weights).layers, self.lr, self.loss_fn, last=self.last.copy(weights))