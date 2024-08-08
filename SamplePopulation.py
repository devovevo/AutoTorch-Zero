import torch
from tqdm import tqdm

from SampleLayers.SampleSequentialLayer import SampleSequential

from NSGAII import nsga_ii
from Selection import tournament_selection

class SamplePopulation():
    def __init__(self, supernet, max_pop, init_pool=None):
        self.supernet = supernet

        self.max_pop = max_pop
        self.pool = init_pool[:max_pop, :supernet.path_length] if init_pool is not None else None

        self.stats = torch.ones((max_pop, 2)) * torch.inf

    def initialize_randomly(self):
        self.pool = torch.randint(high=len(self.supernet.layers), size=(self.max_pop, self.supernet.max_depth))
        self.stats = torch.ones((self.max_pop, 2)) * torch.inf

    def gen_paths(self):
        return torch.stack([torch.randperm(len(self.supernet.layers)) for _ in range(self.supernet.max_depth)], axis=1)
    
    def train_fair_dataset(self, x_train, y_train, epochs=100, batch_size=32):
        self.supernet.train_fair_dataset(x_train, y_train, self.gen_paths, epochs, batch_size)

    def train_fair_dataloader(self, dataloader, epochs=100, device=torch.device('cpu')):
        self.supernet.train_fair_dataloader(dataloader, self.gen_paths, epochs, device)

    def evaluate_pool_dataloader(self, dataloader, device=torch.device('cpu')):
        for nn in tqdm(range(self.max_pop), desc="Evaluating Pool"):
            self.stats[nn, 0] = self.supernet.evaluate_dataloader(dataloader, device, self.pool[nn])
            self.stats[nn, 1] = self.supernet.flops(self.pool[nn])

    def evaluate_pool_dataset(self, x_test, y_test):
        for nn in tqdm(range(self.max_pop), desc="Evaluating Pool"):
            self.stats[nn, 0] = self.supernet.evaluate_dataset(x_test, y_test, self.pool[nn])
            self.stats[nn, 1] = self.supernet.flops(self.pool[nn])

    def mutate_path(self, path):
        rand_index = torch.randint(high=len(path), size=())
        path[rand_index] = torch.randint(high=len(self.supernet.layers), size=())

    def gen_children(self, out, k, criteria):
        for nn in tqdm(range(self.max_pop), desc="Generating Children"):
            parent = tournament_selection(self.pool, k, criteria)

            child = parent.clone().detach()
            self.mutate_path(child)

            if nn < len(out):
                out[nn] = child
            else:
                out.append(child)

class CompoundSamplePopulation(SamplePopulation):
    def initialize_randomly(self):
        self.pool = torch.empty((self.max_pop, self.supernet.path_length), dtype=torch.int64)
        self.pool[:, 0] = torch.randint(high=len(self.supernet.compound_layers), size=(self.max_pop,))
        self.pool[:, 1:] = torch.randint(high=len(self.supernet.layers), size=(self.max_pop, self.supernet.max_depth))

        self.stats = torch.ones((self.max_pop, 2)) * torch.inf

    def gen_paths(self):
        paths = torch.empty(size=(len(self.supernet.layers) * len(self.supernet.compound_layers), self.supernet.path_length), dtype=torch.int64)
        layer_perms = super().gen_paths()

        for l in torch.arange(start=0, end=len(self.supernet.compound_layers) * len(self.supernet.layers), step=len(self.supernet.layers)):
            paths[l:l + len(self.supernet.layers), 0] = torch.ones(len(self.supernet.layers)) * l // len(self.supernet.layers)
            paths[l:l + len(self.supernet.layers), 1:] = layer_perms

        return paths

    def mutate_path(self, path):
        rand_index = torch.randint(high=len(path), size=())

        if rand_index == 0:
            path[rand_index] = torch.randint(high=len(self.supernet.compound_layers), size=())
        else:
            path[rand_index] = torch.randint(high=len(self.supernet.layers), size=())    

class ChainSamplePopulation(CompoundSamplePopulation):
    def initialize_randomly(self):
        self.pool = torch.empty((self.max_pop, self.supernet.chain_length * (self.supernet.max_depth + 1)), dtype=torch.int64)

        for c in range(self.supernet.chain_length):
            self.pool[:, c * (self.supernet.max_depth + 1)] = torch.randint(0, len(self.supernet.compound_layers), size=(self.max_pop,))
            self.pool[:, c * (self.supernet.max_depth + 1) + 1:(c + 1) * (self.supernet.max_depth + 1)] = torch.randint(0, len(self.supernet.layers), size=(self.max_pop, self.supernet.max_depth))

    def gen_paths(self):
        return torch.cat([super().gen_paths() for _ in range(self.supernet.chain_length)], dim=1)

    def mutate_path(self, path):
        rand_index = torch.randint(high=len(path), size=())

        if rand_index % (self.supernet.max_depth + 1) == 0:
            path[rand_index] = torch.randint(high=len(self.supernet.compound_layers), size=())
        else:
            path[rand_index] = torch.randint(high=len(self.supernet.layers), size=())

class NSGIISamplePopulation(SamplePopulation):
    def __init__(self, supernet, max_pop, init_pool = None):
        super().__init__(supernet, max_pop, init_pool)

        self.children = torch.zeros((self.max_pop, supernet.path_length), dtype=torch.int64)
        self.children_stats = torch.ones((self.max_pop, 2)) * torch.inf

    def gen_children(self, k):
        super().gen_children(self.children, k, torch.arange(self.max_pop))
        self.children_stats = torch.ones((self.max_pop, 2)) * torch.inf

    def evaluate_children_dataloader(self, dataloader, device=torch.device('cpu')):
        for nn in tqdm(range(self.max_pop), desc="Evaluating Children"):
            self.children_stats[nn, 0] = self.supernet.evaluate_dataloader(dataloader, device, self.children[nn])
            self.children_stats[nn, 1] = self.supernet.flops(self.children[nn])

    def evaluate_children_dataset(self, x_test, y_test):
        for nn in tqdm(range(self.max_pop), desc="Evaluating Children"):
            self.children_stats[nn, 0] = self.supernet.evaluate_dataset(x_test, y_test, self.children[nn])
            self.children_stats[nn, 1] = self.supernet.flops(self.children[nn])

    def step(self, percentile=0.75):
        combined_pool = torch.concatenate(tensors=[self.pool, self.children])
        combined_stats = torch.concatenate(tensors=[self.stats, self.children_stats])

        sorted_acc_indices = torch.argsort(combined_stats[:, 0])
        dropped_acc_indices = sorted_acc_indices[:int(percentile * len(sorted_acc_indices))]

        sorted_pool = combined_pool[dropped_acc_indices]
        sorted_stats = combined_stats[dropped_acc_indices]

        new_pop_indices = nsga_ii(self.max_pop, sorted_stats)

        self.pool = sorted_pool[new_pop_indices]
        self.stats = sorted_stats[new_pop_indices]

class NSGIICompoundSamplePopulation(CompoundSamplePopulation, NSGIISamplePopulation):
    pass

class NSGIIChainSamplePopulation(ChainSamplePopulation, NSGIISamplePopulation):
    pass