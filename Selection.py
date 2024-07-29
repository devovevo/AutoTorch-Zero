import numpy as np

def tournament_selection(pool, k, criteria):
        indices = np.random.choice(len(pool), k)
        return pool[indices[np.argmin(criteria[indices])]]