"""Initial-state generators for the stable set solvers.
 
Random-order greedy
-------------------
Both initializers in this module use the same algorithm:
 
1. Visit the vertices in a uniformly random order.
2. When a vertex that has not been removed is reached, add it to the set.
   Then remove it and all of its neighbours.
 
The result is always a *maximal* stable set, which has two properties:
 
* it is stable: no two chosen vertices are adjacent;
* it is maximal: every vertex that was not chosen has a chosen neighbour.
"""
import numpy as np
from numba import njit

def random_order_init(graph, rng):
    """Build a random maximal stable set with a random-order greedy pass.
 
    This is the pure-Python version.
 
    Parameters
    ----------
    graph : CSRGraph
        The input graph.
    num_states : int
        Number of values a vertex can take. Unused; accepted only so the
        function matches the ``InitFn`` signature.
    rng : np.random.Generator
        Random generator. Consumed by exactly one call to
        ``rng.permutation(graph.n)``.
 
    Returns
    -------
    np.ndarray of int64, shape (n,)
        Indicator vector of a maximal stable set.
    """
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
 
 
def random_order_init_jit(graph, rng):
    """Build a random maximal stable set, using Numba.
 
    Returns exactly the same array as :func:`random_order_init` when given a
    generator in the same state, but runs faster on large graphs.
 
    Parameters
    ----------
    graph : CSRGraph
        The input graph.
    rng : np.random.Generator
        Random generator. Consumed by exactly one call to
        ``rng.permutation(graph.n)``.
 
    Returns
    -------
    np.ndarray of int64, shape (n,)
        Indicator vector of a maximal stable set.
    """
    offsets = graph.offsets.astype(np.int64)
    nbrs = graph.nbrs.astype(np.int64)
    order = rng.permutation(graph.n).astype(np.int64)
    return _random_order_core(offsets, nbrs, order)