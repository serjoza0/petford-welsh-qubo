
import time
import numpy as np
from numba import njit
from .solver import BSpec


@njit(cache=True)
def _run_core(state, neighbor_sum, is_conflicted, degree, offsets, nbrs,
              A, B, temps, candidate_draws, accept_draws, max_iters, target):
    n = state.shape[0]
    num_conflicted = 0
    for v in range(n):
        if is_conflicted[v]:
            num_conflicted += 1
    set_size = 0
    for v in range(n):
        set_size += state[v]

    best_value = -1
    best_step = -1
    best_state = state.copy()
    if num_conflicted == 0:
        best_value = set_size
        best_step = 0

    for it in range(max_iters):
        u = candidate_draws[it]
        x_u = state[u]
        s = neighbor_sum[u]
        deg = degree[u]
        abs_term = s if x_u == 0 else deg - s
        # X0 = A * x_u - B * abs_term
        # X1 = B * s - A * (1 - x_u)
        X0 = 0
        X1 = B * s - A
        # X0 = A * x_u - B * s * x_u
        # X1 = B * s - B * s * x_u - A * (1 - x_u)

        temp = temps[it]
        m = X0 if X0 < X1 else X1
        w0 = np.exp((m - X0) / temp)
        w1 = np.exp((m - X1) / temp)
        p1 = w1 / (w0 + w1)
        new_val = 1 if accept_draws[it] < p1 else 0

        if new_val != x_u:
            delta = new_val - x_u
            state[u] = new_val
            set_size += delta
            for idx in range(offsets[u], offsets[u + 1]):
                v = nbrs[idx]
                neighbor_sum[v] += delta
                now = (state[v] == 1) and (neighbor_sum[v] > 0)
                if now != is_conflicted[v]:
                    is_conflicted[v] = now
                    num_conflicted += 1 if now else -1
            now_u = (state[u] == 1) and (neighbor_sum[u] > 0)
            if now_u != is_conflicted[u]:
                is_conflicted[u] = now_u
                num_conflicted += 1 if now_u else -1

            if num_conflicted == 0 and set_size > best_value:
                best_value = set_size
                best_step = it + 1
                best_state = state.copy()

                if target >= 0 and set_size >= target:
                    break

    return best_state, best_value, best_step



def petford_welsh_jit(graph, A=1.0, B=2.0, b: BSpec = 4.0, max_iters=1000, rng=None, target=None, init_fn=None):
    t0 = time.perf_counter()

    rng = rng or np.random.default_rng()
    n = graph.n
    degree = np.diff(graph.offsets).astype(np.int64)

    if callable(b):
        temps = 1.0 / np.log(np.asarray([b(it) for it in range(max_iters)], dtype=np.float64))
    elif np.isscalar(b):
        temps = np.full(max_iters, 1.0 / np.log(b))
    else:
        temps = 1.0 / np.log(np.asarray(b, dtype=np.float64))

    candidate_draws = rng.integers(0, n, size=max_iters)
    accept_draws = rng.random(size=max_iters)

    offsets = graph.offsets.astype(np.int64)
    nbrs = graph.nbrs.astype(np.int64)
    target_arg = -1 if target is None else int(target)

    if init_fn is not None:
        state = init_fn(graph, 2, rng).astype(np.int64)
        src = np.repeat(np.arange(n), np.diff(offsets))
        neighbor_sum = np.bincount(src, weights=state[nbrs], minlength=n).astype(np.int64)
        is_conflicted = (state == 1) & (neighbor_sum > 0)
    else:
        state = np.zeros(n, dtype=np.int64)
        neighbor_sum = np.zeros(n, dtype=np.int64)
        is_conflicted = np.zeros(n, dtype=np.bool_)

    best_state, best_value, best_step = _run_core(
        state, neighbor_sum, is_conflicted, degree, offsets, nbrs,
        A, B, temps, candidate_draws, accept_draws, max_iters, target_arg,
    )
    elapsed = time.perf_counter() - t0

    return best_state, best_value, best_step, elapsed