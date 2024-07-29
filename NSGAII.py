
import torch
import numpy as np

from tqdm import tqdm

def fast_non_dominated_sort(values):
    def dominates(p, q):
        corrected_p_values = torch.nan_to_num(values[p], nan=torch.inf)
        corrected_q_values = torch.nan_to_num(values[q], nan=torch.inf)

        return torch.all(corrected_p_values <= corrected_q_values) and torch.any(corrected_p_values < corrected_q_values)

    n = values.shape[0]

    dominated_sets = np.zeros(n, dtype=object)
    dominated_counts = torch.zeros(size=(n,), dtype=torch.int64)

    fronts = []
    Q = []

    for p in tqdm(range(n), desc="Computing dominated sets"):
        dominated_sets[p] = []

        for q in range(n):
            if dominates(p, q):
                dominated_sets[p].append(q)
            elif dominates(q, p):
                dominated_counts[p] += 1
        
        if dominated_counts[p] == 0:
            Q.append(p)

    i = 0

    with tqdm(total=n, desc="Computing fronts") as pbar:
        while(len(Q) > 0):
            fronts.append(Q)
            pbar.update(len(Q))

            Q = []

            for p in fronts[i]:
                for q in dominated_sets[p]:
                    dominated_counts[q] -= 1

                    if dominated_counts[q] == 0:
                        Q.append(q)
            
            i += 1

    return fronts

def crowding_distance_assignment(values):
    n, m = values.shape

    if n <= 2:
        return torch.ones(size=(n,)) * torch.inf

    distances = torch.zeros(size=(n,))

    for objective in range(m):
        objective_values = torch.nan_to_num(values[:, objective], posinf=0, neginf=0)

        sorted_indices = torch.argsort(objective_values)
        sorted_values = objective_values[sorted_indices]

        distances[sorted_indices[0]] = torch.inf
        distances[sorted_indices[-1]] = torch.inf

        if sorted_values[-1] == sorted_values[0]:
            continue

        diff = sorted_values[2:] - sorted_values[:-2]
        distances[sorted_indices[1:-1]] += diff / np.nanmax(diff.numpy())
    
    return distances

def nsga_ii(n, values):
    fronts = fast_non_dominated_sort(values)
    new_pop = []

    i = 0

    while len(new_pop) + len(fronts[i]) <= n:
        new_pop.extend(fronts[i])
        i += 1

    if len(new_pop) < n:
        distances = crowding_distance_assignment(values[fronts[i]])
        sorted_indices = torch.argsort(distances)

        new_pop.extend(fronts[i][j] for j in sorted_indices[-(n - len(new_pop)):])

    return new_pop