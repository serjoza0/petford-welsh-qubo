from .graph import CSRGraph, neighbors
from typing import Callable
import numpy as np
import time

InitFn = Callable[[CSRGraph, int, np.random.Generator], np.ndarray]

class MaxStableSetProblem:
    def __init__(self, graph:CSRGraph, A: float = 1.0, B: float = 2.0) -> None:
        # assert B >= A
        self.graph = graph
        self.n = graph.n
        self.num_states = 2
        self.A = A
        self.B = B
        self._all = np.arange(self.n)
        self.degree = np.diff(graph.offsets)
        self.state = np.zeros(self.n, dtype=np.int64)

    def initial_state(self, rng: np.random.Generator, init_fn: InitFn | None = None) -> np.ndarray:
        state = init_fn(self.graph, self.num_states, rng) if init_fn else np.zeros(self.n, dtype=np.int64)
        self.state = state

        src = np.repeat(np.arange(self.n), np.diff(self.graph.offsets))
        self.neighbor_sum = np.bincount(src, weights=state[self.graph.nbrs], minlength=self.n).astype(np.int64)
        self.conflicted = {v for v in range(self.n) if state[v] == 1 and self.neighbor_sum[v] > 0}

        self.best_state: np.ndarray = state.copy()
        self.best_value = -np.inf
        self.best_time = 0.0
        self.best_step = 0
        self._step = 0
        self._set_size = int(state.sum())
        self._start_time = time.perf_counter()
        self._update_best()
        return state

    def candidate_vertices(self) -> np.ndarray:
        return self._all

    def local_field(self, u: int):
        x_u = self.state.item(u)
        s = self.neighbor_sum.item(u)
        deg = self.degree.item(u)
        abs_term = s if x_u == 0 else deg - s
        X0 = self.A * x_u - self.B * abs_term
        X1 = self.B * s - self.A * (1 - x_u)
        return (X0, X1)

    def apply(self, u: int, new_val: int) -> None:
        self._step += 1
        old_val = self.state[u]
        if new_val == old_val:
            return
        self.state[u] = new_val
        delta = new_val - old_val
        self._set_size += delta
        for v in neighbors(self.graph, u):
            self.neighbor_sum[v] += delta
            self._refresh_conflict(v)
        self._refresh_conflict(u)
        self._update_best()

    def energy(self) -> float:
        return float(-self.A * self.state.sum() + self.B * 0.5 * np.dot(self.state, self.neighbor_sum))

    def _refresh_conflict(self, v: int) -> None:
        if self.state[v] == 1 and self.neighbor_sum[v] > 0:
            self.conflicted.add(v)
        else:
            self.conflicted.discard(v)

    def is_feasible(self) -> bool:
        return len(self.conflicted) == 0

    def objective(self) -> float:
        return self._set_size

    def _update_best(self) -> None:
        if self.is_feasible():
            value = self.objective()
            if value > self.best_value:
                self.best_value = value
                self.best_state = self.state.copy()
                self.best_time = time.perf_counter() - self._start_time
                self.best_step = self._step

    def convert_to_feasible(self) -> np.ndarray:
        state = self.state.copy()
        graph = self.graph
        n = graph.n

        neighbor_sum = np.zeros(n, dtype=np.int64)
        for v in range(n):
            if state[v]:
                neighbor_sum[v] = int(state[neighbors(graph, v)].sum())

        conflicted = {v for v in range(n) if state[v] == 1 and neighbor_sum[v] > 0}

        while conflicted:
            v = max(conflicted, key=lambda x: neighbor_sum[x])
            state[v] = 0
            conflicted.discard(v)
            for u in neighbors(graph, v):
                if state[u] == 1:
                    neighbor_sum[u] -= 1
                    if neighbor_sum[u] == 0:
                        conflicted.discard(u)

        return state