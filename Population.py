import torch

class Population():
    def __init__(self, max_pop, init_pop=[]):
        self.max_pop = max_pop
        self.pop = init_pop[:max_pop]

    def add_random(self, num):
        for i in range(min(num, self.max_pop - len(self.pop))):
            