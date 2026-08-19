from typing import Protocol, List
import numpy as np
from .problems import InitFn
from .graph import CSRGraph

class Problem(Protocol):
    n: int
    num_states: int
    state: np.ndarray
    best_state: np.ndarray | None
    best_value: float
    graph: "CSRGraph"
    def initial_state(self, rng: np.random.Generator, init_fn: InitFn | None = None) -> np.ndarray: ...
    def candidate_vertices(self) -> np.ndarray: ...
    def local_field(self, u: int) -> np.ndarray: ...
    def apply(self, u: int, new_val: int) -> None: ...
    # def objective(self, state: np.ndarray) -> float: ...


def boltzmann_choice(field: np.ndarray, b: float, rng: np.random.Generator) -> int:
    # b = np.exp(1 / temp)
    # temp = 1 / np.log(b)
    temp = 1 / np.log(b)
    exponent = - field / temp
    exponent = exponent - np.max(exponent)
    weights = np.exp(exponent)
    probs = weights / weights.sum()
    return rng.choice(len(field), p=probs)


def petford_welsh(
        problem: Problem,
        b: float = 4.0,
        max_iters: int = 1000,
        rng: np.random.Generator | None = None,
        init_fn: InitFn | None = None,
        record_every: int = 1,
        target: float | None = None
):
    rng = rng or np.random.default_rng()
    problem.initial_state(rng, init_fn=init_fn)
    history: List[np.ndarray] = []

    for it in range(max_iters):
        candidates = problem.candidate_vertices()
        if len(candidates) == 0:
            print(f"No candidates left at iteration {it}, terminating early.")
            break

        u = rng.choice(candidates)
        field = problem.local_field(u)
        new_val = boltzmann_choice(field, b, rng)
        problem.apply(u, new_val)
        if it % record_every == 0:
            history.append(problem.best_state.copy() if problem.best_state is not None else problem.state.copy())
        if target is not None and problem.best_value >= target:
            break

    return problem.state, history
