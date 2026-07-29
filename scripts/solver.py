from typing import Protocol, List
import numpy as np
from .problems import InitFn
from .graph import CSRGraph

class Problem(Protocol):
    n: int
    num_states: int
    best_state: np.ndarray | None
    graph: "CSRGraph"
    def initial_state(self, rng: np.random.Generator, init_fn: InitFn | None = None) -> np.ndarray: ...
    def candidate_vertices(self, state: np.ndarray) -> np.ndarray: ...
    def local_field(self, state: np.ndarray, u: int) -> np.ndarray: ...
    def apply(self, state: np.ndarray, u: int, new_val: int) -> None: ...
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
        record_every: int = 1
):
    rng = rng or np.random.default_rng()
    state = problem.initial_state(rng, init_fn=init_fn)
    # print(f"Initial state: {state}")
    history: List[np.ndarray] = []

    for it in range(max_iters):
        candidates = problem.candidate_vertices(state)
        # print(f"Iteration {it}: candidates = {candidates}, state = {state}")
        if len(candidates) == 0:
            print(f"No candidates left at iteration {it}, terminating early.")
            break

        u = rng.choice(candidates)
        field = problem.local_field(state, u)
        new_val = boltzmann_choice(field, b, rng)
        # print(f"Selected vertex {u}, with value {state[u]}, field = {field}, new value = {new_val}")
        problem.apply(state, u, new_val)
        if it % record_every == 0:
            # print(f"Recording state at iteration {it}: {state}")
            history.append(state.copy())

    # print(f"History: {history}")
    return state, history