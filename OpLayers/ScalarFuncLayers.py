import torch

from SimpleLayer import Simple

class Exp(Simple):
    def forward(self, x):
        return torch.exp(x)
    
class Log(Simple):
    def forward(self, x):
        return torch.log(x)
    
class Reciprocal(Simple):
    def forward(self, x):
        return 1 / x
    
class Identity(Simple):
    def forward(self, x):
        return x
    
    def flops(self):
        return 0
    
class Abs(Simple):
    def forward(self, x):
        return torch.abs(x)
    
class Sin(Simple):
    def forward(self, x):
        return torch.sin(x)