import torch

from SimpleLayer import Simple

class Add(Simple):
    def forward(self, x, y):
        return x + y
    
class Multiply(Simple):
    def forward(self, x, y):
        return x * y
    
class Maximum(Simple):
    def forward(self, x, y):
        return torch.maximum(x, y)
    
class Power(Simple):
    def forward(self, x, y):
        return torch.pow(torch.abs(x), y)