import torch
from tqdm import tqdm

class Trainable():
    def __init__(self, lr=0.001, loss_fn=torch.nn.MSELoss()):
        self.loss_fn = loss_fn
        
        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def train_dataloader(self, dataloader, epochs=100, device=torch.device('cpu'), path=None):
        for epoch in tqdm(range(epochs), desc='Training'):
            for i, data in enumerate(dataloader):
                inputs, labels = data

                self.optimizer.zero_grad()

                if path is None:
                    pred = self.forward(inputs.to(device))
                else:
                    pred = self.forward(path, inputs.to(device))
                
                loss = self.loss_fn(pred, labels.to(device))

                if pred.requires_grad:
                    loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, error_if_nonfinite=True)
                    self.optimizer.step()
                except RuntimeError:
                    continue

    def train_dataset(self, x_train, y_train, epochs=100, batch_size=32, path=None):
        for epoch in tqdm(range(epochs), desc='Training'):
            for i in range(0, len(x_train), batch_size):
                inputs = x_train[i:i+batch_size]
                labels = y_train[i:i+batch_size]

                self.optimizer.zero_grad()

                if path is None:
                    pred = self.forward(inputs)
                else:
                    pred = self.forward(path, inputs)
                
                loss = self.loss_fn(pred, labels)

                if pred.requires_grad:
                    loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, error_if_nonfinite=True)
                    self.optimizer.step()
                except RuntimeError:
                    continue

    def evaluate_dataloader(self, dataloader, device=torch.device('cpu'), path=None):
        running_loss = 0

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, labels = data

                if path is None:
                    pred = self.forward(inputs.to(device))
                else:
                    pred = self.forward(path, inputs.to(device))
                
                running_loss += self.loss_fn(pred, labels.to(device)).item()

        return running_loss / len(dataloader)
    
    def evaluate_dataset(self, x_test, y_test, path=None):
        with torch.no_grad():
            if path is None:
                pred = self.forward(x_test)
            else:
                pred = self.forward(path, x_test)

            loss = self.loss_fn(pred, y_test).item()

        return loss