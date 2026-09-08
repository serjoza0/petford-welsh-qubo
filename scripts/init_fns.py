import heapq
import numpy as np
from numba import njit

from .graph import CSRGraph, neighbors

def random_order_init(graph, num_states, rng):
    n = graph.n
    offsets = graph.offsets
    nbrs = graph.nbrs
    order = rng.permutation(n)
    state = np.zeros(n, dtype=np.int64)
    removed = np.zeros(n, dtype=bool)
    for v in order:
        if not removed[v]:
            state[v] = 1
            removed[v] = True
            for idx in range(offsets[v], offsets[v + 1]):
                u = nbrs[idx]
                removed[u] = True
    return state
 
 
@njit(cache=True)
def _random_order_core(offsets, nbrs, order):
    n = order.shape[0]
    state = np.zeros(n, dtype=np.int64)
    removed = np.zeros(n, dtype=np.bool_)
    for idx in range(n):
        v = order[idx]
        if not removed[v]:
            state[v] = 1
            removed[v] = True
            for idx2 in range(offsets[v], offsets[v + 1]):
                u = nbrs[idx2]
                removed[u] = True
    return state
 
 
def random_order_init_jit(graph, num_states, rng):
    offsets = graph.offsets.astype(np.int64)
    nbrs = graph.nbrs.astype(np.int64)
    order = rng.permutation(graph.n).astype(np.int64)
    return _random_order_core(offsets, nbrs, order)