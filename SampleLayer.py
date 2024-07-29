import torch
from tqdm import tqdm

from TrainableLayer import Trainable

class Sample(Trainable):
    def __init__(self, lr=0.001, loss_fn=torch.nn.MSELoss()):
        Trainable.__init__(self, lr, loss_fn)

    def train_fair_dataloader(self, dataloader, paths_fn, epochs=100, device=torch.device('cpu')):
        for epoch in tqdm(range(epochs), desc='Training'):
            for i, data in enumerate(dataloader):
                inputs, labels = data

                self.optimizer.zero_grad()

                paths = paths_fn()

                for path in paths:
                    pred = self.forward(path, inputs.to(device))
                    loss = self.loss_fn(pred, labels.to(device))

                    if pred.requires_grad:
                        loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, error_if_nonfinite=True)
                    self.optimizer.step()
                except RuntimeError:
                    continue

    def train_fair_dataset(self, x_train, y_train, paths_fn, epochs=100, batch_size=32):
        for epoch in tqdm(range(epochs), desc='Training'):
            for i in range(0, len(x_train), batch_size):
                inputs = x_train[i:i+batch_size]
                labels = y_train[i:i+batch_size]

                self.optimizer.zero_grad()

                paths = paths_fn()

                for path in paths:
                    pred = self.forward(path, inputs)
                    loss = self.loss_fn(pred, labels)

                    if pred.requires_grad:
                        loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, error_if_nonfinite=True)
                    self.optimizer.step()
                except RuntimeError:
                    continue