import torch

from tqdm import tqdm

from EvolvedNN import EvolvedNN

from NSGAII import nsga_ii
from Selection import tournament_selection

class EvolvedNNPopulation():
    def __init__(self, input_dim, output_dim, max_pop, init_pool=[], device=torch.device('cpu')):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.device = device

        self.max_pop = max_pop
        self.pool = torch.nn.ModuleList(nn for nn in init_pool[:max_pop]).to(device)

        self.stats = torch.ones((max_pop, 2)) * torch.inf

    def initialize_randomly(self, layers, init_depth, lr=0.001, loss_fn=torch.nn.MSELoss()):
        for nn in tqdm(range(self.max_pop), desc='Initializing Randomly'):
            new_nn = EvolvedNN(self.input_dim, self.output_dim, lr=lr, loss_fn=loss_fn).to(self.device)

            for d in range(init_depth):
                new_nn.mutate(layers, p=[1., 0., 0.])

            new_nn = new_nn.to(self.device)

            if nn < len(self.pool):
                self.pool[nn] = new_nn
            else:
                self.pool.append(new_nn)

        self.stats = torch.ones((self.max_pop, 2)) * torch.inf

    def train_pool_dataloader(self, dataloader, epochs=100):
        for nn in tqdm(range(len(self.pool)), desc='Training Pool'):
            self.pool[nn].train_dataloader(dataloader, epochs, self.device)

    def train_pool_dataset(self, x_train, y_train, epochs=100, batch_size=32):
        for nn in tqdm(range(len(self.pool)), desc='Training Pool'):
            self.pool[nn].train_dataset(x_train, y_train, epochs, batch_size)

    def evaluate_pool_dataloader(self, dataloader):
        for nn in tqdm(range(len(self.pool)), desc='Evaluating Pool'):
            self.stats[nn][0] = self.pool[nn].evaluate_dataloader(dataloader, self.device)
            self.stats[nn][1] = self.pool[nn].flops()
    
    def evaluate_pool_dataset(self, x_test, y_test):
        for nn in tqdm(range(len(self.pool)), desc='Evaluating Pool'):
            self.stats[nn][0] = self.pool[nn].evaluate_dataset(x_test, y_test)
            self.stats[nn][1] = self.pool[nn].flops()

    def gen_children(self, out, k, layers, weights=False, p=[0.3, 0.3, 0.4], criteria=None):
        if criteria is None:
            criteria = self.losses

        for nn in range(self.max_pop):
            parent = tournament_selection(self.pool, k, criteria)

            child = parent.copy(weights).to(self.device)
            child.mutate(layers, p)

            child = child.to(self.device)

            if nn < len(out):
                out[nn] = child
            else:
                out.append(child)

class NSGSIIEvolvedNNPopulation(EvolvedNNPopulation):
    def __init__(self, input_dim, output_dim, max_pop, init_pool=[], device=torch.device('cpu')):
        super().__init__(input_dim, output_dim, max_pop, init_pool, device)

        self.children = torch.nn.ModuleList([]).to(device)
        self.children_stats = torch.ones((max_pop, 2)) * torch.inf

    def gen_children(self, k, layers, weights=False, p=[0.3, 0.3, 0.4]):
        super().gen_children(self.children, k, layers, weights, p, torch.arange(self.max_pop))
        self.children_stats = torch.ones((self.max_pop, 2)) * torch.inf

    def train_children_dataloader(self, dataloader, epochs=100):
        for nn in tqdm(range(len(self.children)), desc='Training Children'):
            self.children[nn].train_dataloader(dataloader, epochs, self.device)

    def train_children_dataset(self, x_train, y_train, epochs=100, batch_size=32):
        for nn in tqdm(range(len(self.children)), desc='Training Children'):
            self.children[nn].train_dataset(x_train, y_train, epochs, batch_size)

    def evaluate_children_dataloader(self, dataloader):
        for nn in tqdm(range(len(self.children)), desc='Evaluating Children'):
            self.children_stats[nn][0] = self.children[nn].evaluate_dataloader(dataloader, self.device)
            self.children_stats[nn][1] = self.children[nn].flops()        

    def evaluate_children_dataset(self, x_test, y_test):
        for nn in tqdm(range(len(self.children)), desc='Evaluating Children'):
            self.children_stats[nn][0] = self.children[nn].evaluate_dataset(x_test, y_test)
            self.children_stats[nn][1] = self.children[nn].flops()

    def step(self, percentile=0.75):
        combined_pool = self.pool + self.children
        combined_stats = torch.concatenate([self.stats, self.children_stats], axis=0)

        sorted_acc_indices = torch.argsort(combined_stats[:, 0])
        dropped_acc_indices = sorted_acc_indices[:int(percentile * len(sorted_acc_indices))]

        sorted_pool = [combined_pool[i] for i in dropped_acc_indices]
        sorted_stats = combined_stats[dropped_acc_indices]

        new_pop_indices = nsga_ii(self.max_pop, sorted_stats)
        
        self.pool = torch.nn.ModuleList([sorted_pool[i] for i in new_pop_indices]).to(self.device)
        self.stats = sorted_stats[new_pop_indices]