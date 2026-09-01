from dataclasses import dataclass, field
from typing import Literal, Callable
import numpy as np
from time import perf_counter
import os
import glob
import csv

from .graph import CSRGraph
from .problems import MaxStableSetProblem
from .solver import petford_welsh, BSpec
from .solver_jit import petford_welsh_jit
SolverName = Literal["python", "jit"]

@dataclass
class AttemptResult:
    seed: int
    best_value: int
    best_step: int
    elapsed: float

@dataclass
class ExperimentResult:
    instance: str
    n: int
    m: int
    A: float
    B: float
    b_repr: str
    max_iters: int
    num_attempts: int
    solver: str
    attempts: list[AttemptResult] = field(default_factory=list)
    known_alpha: int | None = None
    total_time: float = 0.0

    @property
    def sizes(self):
        return [a.best_value for a in self.attempts]
    @property
    def best(self) -> int:
        return max(self.sizes) if self.attempts else 0
    @property
    def mean(self) -> float:
        return float(np.mean(self.sizes)) if self.attempts else 0
    @property
    def std(self) -> float:
        return float(np.std(self.sizes)) if self.attempts else 0.0
    @property
    def gap(self):
        return (self.known_alpha - self.best) if self.known_alpha is not None else None

    def to_row(self) -> dict:
        return {
            "instance": self.instance, "n": self.n, "m": self.m,
            "A": self.A, "B": self.B, "b": self.b_repr,
            "max_iters": self.max_iters, "num_attempts": self.num_attempts,
            "best": self.best, "mean": self.mean, "std": self.std,
            "time_sec": self.total_time,
            "known_alpha": self.known_alpha if self.known_alpha is not None else "",
            "gap": self.gap if self.gap is not None else "",
        }

def _b_repr(b: BSpec) -> str:
    if callable(b):
        return "fn"
    if np.isscalar(b):
        return repr(b)
    arr = np.asarray(b)
    return f"array(len={len(arr)}, {arr[0]:.3g}->{arr[-1]:.3g})"


def _run_attempt_python(graph: CSRGraph, A: float, B: float, b: BSpec, max_iters: int,
                        seed: int, target: float | None, init_fn=None) -> AttemptResult:
    rng = np.random.default_rng(seed)
    problem = MaxStableSetProblem(graph, A, B)
    t0 = perf_counter()
    petford_welsh(problem, b=b, max_iters=max_iters, rng=rng, target=target, init_fn=init_fn)
    elapsed = perf_counter() - t0
    return AttemptResult(seed=seed, best_value=int(problem.best_value), best_step=problem.best_step, elapsed=elapsed)

def _run_attempt_jit(graph: CSRGraph, A: float, B: float, b: BSpec, max_iters: int,
                      seed: int, target: "float | None", init_fn=None) -> AttemptResult:
    rng = np.random.default_rng(seed)
    _, best_value, best_step, elapsed = petford_welsh_jit(
        graph, A=A, B=B, b=b, max_iters=max_iters, rng=rng, target=target, init_fn=init_fn
    )
    return AttemptResult(seed=seed, best_value=int(best_value), best_step=int(best_step), elapsed=elapsed)

_SOLVERS: dict = {
    "python": _run_attempt_python,
    "jit": _run_attempt_jit
}

def _load_graph(graph_or_path: CSRGraph | str, name: str | None = None) -> CSRGraph:
    if isinstance(graph_or_path, CSRGraph):
        return graph_or_path
    path = graph_or_path
    inferred_name = name or os.path.basename(path).replace("_stable_set_edge_list.txt", "")
    return CSRGraph.from_edge_list_file(path, name=inferred_name)


def run_multi_start(
    graph: CSRGraph | str,
    *,
    solver: SolverName = "jit",
    A: float = 1.0,
    B: float = 2.0,
    b: float = 4.0,
    max_iters: int = 100000,
    num_attempts: int = 10,
    seed: int = 0,
    known_alpha: int | None = None,
    target: float | None = None,
    init_fn = None,
) -> ExperimentResult:
    if solver not in _SOLVERS:
        raise ValueError("Unknown solver")
    
    run_attempt = _SOLVERS[solver]
    g = _load_graph(graph)

    result = ExperimentResult(
        instance=g.name, n=g.n, m=g.m,
        A=A, B=B, b_repr=_b_repr(b), max_iters=max_iters, num_attempts=num_attempts,
        solver=solver, known_alpha=known_alpha,
    )

    t0 = perf_counter()
    for i in range(num_attempts):
        result.attempts.append(run_attempt(g, A, B, b, max_iters, seed+i, target, init_fn))
    result.total_time = perf_counter() - t0
    return result

def run_all_instances(
        instances_dir: str,
        *,
        known_alpha: dict[str, int] | None = None,
        name_filter: Callable[[str], bool] | None = None,
        console_output: bool = True,
        **run_kwargs,
) -> list:
    known_alpha = known_alpha or {}
    paths = sorted(glob.glob(os.path.join(instances_dir, "*.txt")))
    if name_filter:
        paths = [p for p in paths if name_filter(os.path.basename(p))]

    results = []
    for path in paths:
        name = os.path.basename(path).replace("_stable_set_edge_list.txt", "")
        result = run_multi_start(path, known_alpha=known_alpha.get(name), **run_kwargs)
        results.append(result)
        if console_output:
            gap_str = result.gap if result.gap is not None else "?"
            print(f"[{result.solver:6s}] {result.instance:35s} n={result.n:5d} m={result.m:7d} "
                  f"best={result.best:4d} mean={result.mean:7.2f} std={result.std:5.2f} "
                  f"gap={gap_str} time={result.total_time:6.1f}s")
    return results

def save_results_csv(results: list, path: str) -> None:
    if not results:
        return
    rows = [r.to_row() for r in results]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def save_attempts_csv(results: list, path: str) -> None:
    rows = []
    for r in results:
        for a in r.attempts:
            rows.append({
                "instance": r.instance, "solver": r.solver, "seed": a.seed,
                "best_value": a.best_value, "best_step": a.best_step, "elapsed": a.elapsed,
                "known_alpha": r.known_alpha if r.known_alpha is not None else "",
            })
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)