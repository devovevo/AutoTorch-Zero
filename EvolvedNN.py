import torch

from CompoundLayers.SequentialLayer import Sequential

class EvolvedNN(Sequential):
    def __init__(self, dim, layers=[], optimizer=None, lr=0.001, loss=torch.nn.MSELoss()):
        super().__init__(dim, layers)

        self.lr = lr
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(self.layers.parameters(), lr=lr)

        self.loss = loss

    def train(self, x, y, epochs=100, batch_size=32):
        for epoch in range(epochs):
            for i in range(0, len(x), batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                self.optimizer.zero_grad()

                self.loss = self.loss(self(x_batch), y_batch)
                self.loss.backward()
                
                self.optimizer.step()
    
    def num_layers(self):
        return super().num_layers()
    
    def mutate(self, layers, p_add=0.2, p_remove=0.4, p_replace=0.4):
        num_layers = self.num_layers()

        if num_layers == 0:
            return

        index = torch.randint(num_layers, (1,)).item()
        cur_index = 0

        for layer in self.layers:
            if cur_index == index:
                