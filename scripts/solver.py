from typing import Protocol, List
import numpy as np
from .problems import InitFn
from .graph import CSRGraph
from math import log, exp
from typing import Callable

class Problem(Protocol):
    n: int
    num_states: int
    state: np.ndarray
    best_state: np.ndarray
    best_value: float
    graph: "CSRGraph"
    def initial_state(self, rng: np.random.Generator, init_fn: InitFn | None = None) -> np.ndarray: ...
    def candidate_vertices(self) -> np.ndarray: ...
    def local_field(self, u: int) -> tuple: ...
    def apply(self, u: int, new_val: int) -> None: ...
    # def objective(self, state: np.ndarray) -> float: ...


def boltzmann_choice(field, b: float, u01: float) -> int:
    # b = np.exp(1 / temp)
    # temp = 1 / np.log(b)
    temp = 1 / log(b)
    x0, x1 = field
    # m is to avoid overflow in exp
    m = x0 if x0 < x1 else x1
    w0 = exp((m - x0) / temp)
    w1 = exp((m - x1) / temp)
    p1 = w1 / (w0 + w1)
    return 1 if u01 < p1 else 0

BSpec = float | np.ndarray | Callable[[int], float]

def petford_welsh(
        problem: Problem,
        b: BSpec = 4.0,
        max_iters: int = 1000,
        rng: np.random.Generator | None = None,
        init_fn: InitFn | None = None,
        record_every: int = 1,
        target: float | None = None
):
    rng = rng or np.random.default_rng()
    problem.initial_state(rng, init_fn=init_fn)
    history: List[np.ndarray] = []

    n_candidates = len(problem.candidate_vertices())
    candidate_draws = rng.integers(0, n_candidates, size=max_iters)
    access_draws = rng.random(size=max_iters)

    for it in range(max_iters):
        candidates = problem.candidate_vertices()
        if len(candidates) == 0:
            print(f"No candidates left at iteration {it}, terminating early.")
            break

        u = int(candidates[candidate_draws[it]])
        field = problem.local_field(u)
        b_value: float
        if callable(b):
            b_value = b(it)
        elif isinstance(b, np.ndarray):
            b_value = b[it] if it < len(b) else b[-1]
        else:
            b_value = b
        new_val = boltzmann_choice(field, b_value, access_draws[it])
        problem.apply(u, new_val)
        if it % record_every == 0:
            history.append(problem.best_state.copy() if problem.best_state is not None else problem.state.copy())

        if target is not None and problem.best_value >= target:
            break

    return problem.state, history
