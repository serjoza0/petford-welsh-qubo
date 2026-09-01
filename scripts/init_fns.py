import heapq
import numpy as np
from numba import njit

from .graph import CSRGraph, neighbors


def greedy_min_degree_init(graph: CSRGraph, num_states: int, rng: np.random.Generator) -> np.ndarray:
    n = graph.n
    state = np.zeros(n, dtype=np.int64)
    removed = np.zeros(n, dtype=bool)
    degree = np.diff(graph.offsets).astype(np.int64).copy()

    tie = rng.random(n)
    heap = [(int(degree[v]), tie[v], v) for v in range(n)]
    heapq.heapify(heap)

    while heap:
        d, _, v = heapq.heappop(heap)
        if removed[v]:
            continue
        if d != degree[v]:
            heapq.heappush(heap, (int(degree[v]), tie[v], v))
            continue

        state[v] = 1
        removed[v] = True
        for u in neighbors(graph, v):
            if not removed[u]:
                removed[u] = True
                for w in neighbors(graph, u):
                    if not removed[w]:
                        degree[w] -= 1
                        heapq.heappush(heap, (int(degree[w]), tie[w], w))

    return state


@njit(cache=True)
def _heap_push(heap_key, heap_tie, heap_vert, size, key, tie_v, v):
    heap_key[size] = key
    heap_tie[size] = tie_v
    heap_vert[size] = v
    i = size
    size += 1
    while i > 0:
        parent = (i - 1) // 2
        if (heap_key[parent] > heap_key[i]) or (
            heap_key[parent] == heap_key[i] and heap_tie[parent] > heap_tie[i]
        ):
            heap_key[parent], heap_key[i] = heap_key[i], heap_key[parent]
            heap_tie[parent], heap_tie[i] = heap_tie[i], heap_tie[parent]
            heap_vert[parent], heap_vert[i] = heap_vert[i], heap_vert[parent]
            i = parent
        else:
            break
    return size
 
 
@njit(cache=True)
def _heap_pop(heap_key, heap_tie, heap_vert, size):
    key0 = heap_key[0]
    tie0 = heap_tie[0]
    v0 = heap_vert[0]
 
    size -= 1
    heap_key[0] = heap_key[size]
    heap_tie[0] = heap_tie[size]
    heap_vert[0] = heap_vert[size]
 
    i = 0
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < size and (
            (heap_key[left] < heap_key[smallest])
            or (heap_key[left] == heap_key[smallest] and heap_tie[left] < heap_tie[smallest])
        ):
            smallest = left
        if right < size and (
            (heap_key[right] < heap_key[smallest])
            or (heap_key[right] == heap_key[smallest] and heap_tie[right] < heap_tie[smallest])
        ):
            smallest = right
        if smallest == i:
            break
        heap_key[i], heap_key[smallest] = heap_key[smallest], heap_key[i]
        heap_tie[i], heap_tie[smallest] = heap_tie[smallest], heap_tie[i]
        heap_vert[i], heap_vert[smallest] = heap_vert[smallest], heap_vert[i]
        i = smallest
 
    return key0, tie0, v0, size
 
 
@njit(cache=True)
def _greedy_min_degree_core(offsets, nbrs, degree0, tie):
    n = degree0.shape[0]
    state = np.zeros(n, dtype=np.int64)
    removed = np.zeros(n, dtype=np.bool_)
    degree = degree0.copy()
 
    capacity = n + nbrs.shape[0]
    heap_key = np.empty(capacity, dtype=np.int64)
    heap_tie = np.empty(capacity, dtype=np.float64)
    heap_vert = np.empty(capacity, dtype=np.int64)
    size = 0
 
    for v in range(n):
        size = _heap_push(heap_key, heap_tie, heap_vert, size, degree[v], tie[v], v)
 
    while size > 0:
        key, tie_v, v, size = _heap_pop(heap_key, heap_tie, heap_vert, size)
        if removed[v]:
            continue
        if key != degree[v]:
            size = _heap_push(heap_key, heap_tie, heap_vert, size, degree[v], tie[v], v)
            continue
 
        state[v] = 1
        removed[v] = True
        for idx in range(offsets[v], offsets[v + 1]):
            u = nbrs[idx]
            if not removed[u]:
                removed[u] = True
                for idx2 in range(offsets[u], offsets[u + 1]):
                    w = nbrs[idx2]
                    if not removed[w]:
                        degree[w] -= 1
                        size = _heap_push(heap_key, heap_tie, heap_vert, size, degree[w], tie[w], w)
 
    return state
 
 
def greedy_min_degree_init_jit(graph: CSRGraph, num_states: int, rng: np.random.Generator) -> np.ndarray:
    """
    Drop-in, same-signature replacement for greedy_min_degree_init --
    usable anywhere an init_fn is accepted (run_multi_start, petford_welsh,
    petford_welsh_jit).
    """
    offsets = graph.offsets.astype(np.int64)
    nbrs = graph.nbrs.astype(np.int64)
    degree0 = np.diff(offsets).astype(np.int64)
    tie = rng.random(graph.n)
    return _greedy_min_degree_core(offsets, nbrs, degree0, tie)