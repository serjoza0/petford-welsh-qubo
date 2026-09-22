"""Petford-Welsh solver for Maximum Stable Set.
 
.. math::
 
    E(x) = -A \\sum_v x_v + B \\sum_{\\{u, v\\} \\in E} x_u x_v ,
 
where ``x`` is a 0/1 vector and ``x[v] == 1`` means vertex ``v`` is in the
set. Each chosen vertex lowers the energy by ``A``; each edge with both
endpoints chosen (a *conflict*) raises it by ``B``.
 
Update rule
-----------
One iteration does the following:
 
1. Pick a vertex ``u`` uniformly at random from all vertices.
2. Let ``s`` be the number of neighbours of ``u`` currently in the set.
   Setting ``x_u = 1`` instead of ``0`` changes the energy by
   ``B * s - A``.
3. Set ``x_u = 1`` with probability ::
 
       p1 = 1 / (1 + b ** (B * s - A))
 
   and ``x_u = 0`` otherwise.
 
The parameter ``b > 1`` is the Petford-Welsh base; it corresponds to a
temperature ``T = 1 / ln(b)``. A larger ``b`` means a lower temperature
and greedier moves.
 
With ``u`` free of conflicts (``s = 0``), ``u`` joins or stays in the set
with probability ``b**A / (1 + b**A)``. Each chosen neighbour makes this
``b**B`` times less favourable.
"""
import time
import numpy as np
from numba import njit

@njit(cache=True)
def _run_core(state, neighbor_sum, is_conflicted, offsets, nbrs,
                        A, B, temp, candidate_draws, accept_draws, max_iters):
    """Numba kernel: run the iterations and track the best feasible state.
 
    All random numbers and per-iteration parameters are computed in advance
    by :func:`petford_welsh_jit`. The kernel itself is deterministic.
 
    Parameters
    ----------
    state : np.ndarray of int64, shape (n,)
        Starting 0/1 state. **Modified in place**; holds the final state
        on return.
    neighbor_sum : np.ndarray of int64, shape (n,)
        Number of in-set neighbours of each vertex; must match ``state``.
        Modified in place.
    is_conflicted : np.ndarray of bool, shape (n,)
        ``state[v] == 1 and neighbor_sum[v] > 0`` for each vertex; must
        match ``state``. Modified in place.
    offsets, nbrs : np.ndarray of int64
        CSR structure of the graph.
    A : float
        Reward for each vertex in the set.
    B : float
        Conflict penalty for each iteration.
    temp : float
        Temperature ``1 / ln(b)`` for each iteration.
    candidate_draws : np.ndarray of int64, shape (>= max_iters,)
        Vertex to update at each iteration.
    accept_draws : np.ndarray of float64, shape (>= max_iters,)
        Uniform ``[0, 1)`` numbers used to decide each update.
    max_iters : int
        Number of iterations.

    Returns
    -------
    best_state : np.ndarray of int64, shape (n,)
        Copy of the largest feasible state visited. If none was visited,
        this is a copy of the (infeasible) starting state.
    best_value : int
        Size of ``best_state``, or ``-1`` if no feasible state was visited.
    best_step : int
        Number of iterations completed when ``best_state`` was reached: 0
        if the starting state was best, ``-1`` if no feasible state was
        visited.
    """
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
        X0 = 0
        X1 = B * s - A
 
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
 
    return best_state, best_value, best_step
 
 
def petford_welsh_jit(graph, A=1.0, B=2.0, b=10.0, max_iters=1000, rng=None, init_fn=None):
    """Search for a large stable set with Petford-Welsh.
 
    Parameters
    ----------
    graph : CSRGraph
        The input graph.
    A : float, default 1.0
        Reward for each vertex in the set. Must be a constant.
    B : float, default 2.0
        Conflict penalty. Must be a constant.
    b : float, default 10.0
        Every value must be greater than 1. The corresponding temperature is ``T = 1 / ln(b)``.
    max_iters : int, default 1000
        Number of single-vertex updates.
    rng : np.random.Generator, optional
        Source of randomness. A fresh unseeded generator is used if
        omitted.
    init_fn : InitFn, optional
        Called as ``init_fn(graph, rng)`` to produce the starting
        state; see :mod:`init_fns`. Its output is copied, not modified.
        If omitted, the search starts from the empty set.
 
    Returns
    -------
    best_state : np.ndarray of int64, shape (n,)
        The largest feasible state found.
    best_value : int
        Its size. ``-1`` means no feasible state was found, which can only
        happen with an infeasible ``init_fn``; in that case ``best_state``
        is the infeasible starting state.
    best_step : int
        Number of iterations completed when ``best_state`` was reached: 0
        if the starting state was best, ``-1`` if nothing feasible was
        found.
    elapsed : float
        Wall-clock seconds for the whole call. This includes preparing the
        parameter arrays, drawing the random numbers, running ``init_fn``,
        and, on the first call in a new environment, compiling the kernel.
 
    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> state, size, step, secs = petford_welsh_jit(
    ...     graph, A=1.0, B=2.0, b=4.0, max_iters=100_000,
    ...     rng=rng, init_fn=random_order_init_jit)
    """
    if b <= 1.0:
        raise ValueError(f"b must be > 1, got {b}")

    t0 = time.perf_counter()
 
    rng = rng or np.random.default_rng()
    n = graph.n
    temp = 1.0 / np.log(b)
 
    candidate_draws = rng.integers(0, n, size=max_iters)
    accept_draws = rng.random(size=max_iters)
 
    offsets = graph.offsets.astype(np.int64)
    nbrs = graph.nbrs.astype(np.int64)
 
    if init_fn is not None:
        state = init_fn(graph, rng).astype(np.int64)
        src = np.repeat(np.arange(n), np.diff(offsets))
        neighbor_sum = np.bincount(src, weights=state[nbrs], minlength=n).astype(np.int64)
        is_conflicted = (state == 1) & (neighbor_sum > 0)
    else:
        state = np.zeros(n, dtype=np.int64)
        neighbor_sum = np.zeros(n, dtype=np.int64)
        is_conflicted = np.zeros(n, dtype=np.bool_)
 
    best_state, best_value, best_step = _run_core(
        state, neighbor_sum, is_conflicted, offsets, nbrs,
        A, B, temp, candidate_draws, accept_draws, max_iters,
    )
    elapsed = time.perf_counter() - t0
 
    return best_state, best_value, best_step, elapsed
 