from .graph import CSRGraph, neighbors
from typing import Callable
import numpy as np

InitFn = Callable[[CSRGraph, int, np.random.Generator], np.ndarray]

class GraphColoringProblem:
    def __init__(self, graph: CSRGraph, k: int):
        self.graph = graph
        self.n = graph.n
        self.num_states = k
        self.k = k

    def initial_state(self, rng: np.random.Generator, init_fn: InitFn | None = None) -> np.ndarray:
        state = init_fn(self.graph, self.k, rng) if init_fn else rng.integers(0, self.k, size=self.n)

        self.color_count = np.zeros((self.n, self.k), dtype=np.int32)
        for v in range(self.n):
            for u in neighbors(self.graph, v):
                self.color_count[v, state[u]] += 1
        self.bad_vertices = {v for v in range(self.n) if self.color_count[v, state[v]] > 0}
        self.best_state = None
        self.best_value = -np.inf
        self._update_best(state)
        return state

    def candidate_vertices(self, state: np.ndarray) -> np.ndarray:
        return np.fromiter(self.bad_vertices, dtype=np.int32)

    def local_field(self, state: np.ndarray, u: int) -> np.ndarray:
        return self.color_count[u]

    def apply(self, state: np.ndarray, u: int, new_color: int) -> None:
        old_color = state[u]
        if new_color == old_color:
            return
        state[u] = new_color
        for v in neighbors(self.graph, u):
            self.color_count[v, old_color] -= 1
            self.color_count[v, new_color] += 1
            self._refresh_bad(state, v)
        self._refresh_bad(state, u)
        self._update_best(state)

    def _refresh_bad(self, state: np.ndarray, v: int) -> None:
        if self.color_count[v, state[v]] > 0:
            self.bad_vertices.add(v)
        else:
            self.bad_vertices.discard(v)

    def is_feasible(self, state: np.ndarray) -> bool:
        return len(self.bad_vertices) == 0

    def objective(self, state: np.ndarray) -> float:
        return -len(self.bad_vertices)
    
    def _update_best(self, state: np.ndarray) -> None:
        if self.is_feasible(state):
            value = self.objective(state)
            if value > self.best_value:
                self.best_value = value
                self.best_state = state.copy()


class MaxStableSetProblem:
    def __init__(self, graph:CSRGraph, A: float = 1.0, B: float = 2.0) -> None:
        # assert B >= A
        self.graph = graph
        self.n = graph.n
        self.num_states = 2
        self.A = A
        self.B = B
        self._all = np.arange(self.n)

    def initial_state(self, rng: np.random.Generator, init_fn: InitFn | None = None) -> np.ndarray:
        state = init_fn(self.graph, self.num_states, rng) if init_fn else np.zeros(self.n, dtype=np.int64)
        self.neighbor_sum = np.zeros(self.n, dtype=np.int64)
        for v in range(self.n):
            for u in neighbors(self.graph, v):
                self.neighbor_sum[v] += state[u]
        self.conflicted = {v for v in range(self.n) if state[v] == 1 and self.neighbor_sum[v] > 0}

        self.best_state = None
        self.best_value = -np.inf
        self._update_best(state)
        return state

    def candidate_vertices(self, state: np.ndarray) -> np.ndarray:
        return self._all

    def local_field(self, state: np.ndarray, u: int) -> np.ndarray:
        x_u = state[u]
        s = self.neighbor_sum[u]
        deg = self.graph.offsets[u + 1] - self.graph.offsets[u]
        abs_term = s if x_u == 0 else deg - s
        X0 = self.A * x_u - self.B * abs_term
        X1 = self.B * s - self.A * (1 - x_u)
        return np.array([X0, X1])

    def apply(self, state: np.ndarray, u: int, new_val: int) -> None:
        old_val = state[u]
        if new_val == old_val:
            return
        state[u] = new_val
        delta = new_val - old_val
        for v in neighbors(self.graph, u):
            self.neighbor_sum[v] += delta
            self._refresh_conflict(state, v)
        self._refresh_conflict(state, u)
        self._update_best(state)

    def energy(self, state: np.ndarray) -> float:
        return float(-self.A * state.sum() + self.B * 0.5 * np.dot(state, self.neighbor_sum))

    def _refresh_conflict(self, state: np.ndarray, v: int) -> None:
        if state[v] == 1 and self.neighbor_sum[v] > 0:
            self.conflicted.add(v)
        else:
            self.conflicted.discard(v)

    def is_feasible(self, state: np.ndarray) -> bool:
        return len(self.conflicted) == 0

    def objective(self, state: np.ndarray) -> float:
        return int(state.sum())

    def _update_best(self, state: np.ndarray) -> None:
        if self.is_feasible(state):
            value = self.objective(state)
            if value > self.best_value:
                self.best_value = value
                self.best_state = state.copy()