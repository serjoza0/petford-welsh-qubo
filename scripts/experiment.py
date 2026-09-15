"""Multi-start experiments and CSV reporting.
 
This module runs the solver several times on one instance or on a whole
directory of instances, collects the results, and writes them to CSV.
 
Each instance is run ``num_attempts`` times, with seeds ``seed``,
``seed + 1``, and so on. Every instance uses the same seeds.
 
Instance files
--------------
An instance's name is its file name with
the suffix ``_stable_set_edge_list.txt`` removed. This name is used in
reports and as the key into ``known_alpha``.
"""
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from time import perf_counter
import os
import glob
import csv

from .graph import CSRGraph
from .solver_jit import petford_welsh_jit

@dataclass
class AttemptResult:
    """Outcome of a single solver run.
 
    Attributes
    ----------
    seed : int
        Seed of the ``np.random.Generator`` used for this run.
    best_value : int
        Size of the largest feasible set found.
    elapsed : float
        Wall-clock seconds for the run. For ``"jit"``, this includes setup
        and, on the first run in a fresh environment, compilation (about
        0.5 s).
    """
    seed: int
    best_value: int
    best_step: int
    elapsed: float

@dataclass
class ExperimentResult:
    """All attempts on one instance, with summary statistics.
 
    Created by :func:`run_multi_start`.
 
    Attributes
    ----------
    instance : str
        Instance name.
    n, m : int
        Number of vertices and edges.
    A : float
        Penalty weights passed to the solver. ``B`` is stored as given, so
        it may also be a schedule array or a function.
    B_repr : str
        Short text description of the ``B`` argument; see :func:`_repr`.
    b_repr : str
        Short text description of the ``b`` argument; see
        :func:`_b_repr`.
    max_iters : int
        Iterations per attempt.
    num_attempts : int
        Number of attempts requested.
    attempts : list of AttemptResult
        One entry per attempt, in seed order.
    known_alpha : int or None
        Known stability number of the instance, if provided.
    total_time : float
        Wall-clock seconds for all attempts together. Loading the graph is
        not included.
    """
    instance: str
    n: int
    m: int
    A: float
    B_repr: str
    b_repr: str
    max_iters: int
    num_attempts: int
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
            "A": self.A, "B": self.B_repr, "b": self.b_repr,
            "max_iters": self.max_iters, "num_attempts": self.num_attempts,
            "best": self.best, "mean": self.mean, "std": self.std,
            "time_sec": self.total_time,
            "known_alpha": self.known_alpha if self.known_alpha is not None else "",
            "gap": self.gap if self.gap is not None else "",
        }



def _repr(b: float | np.ndarray | Callable[[int], float]) -> str:
    """Return a short text description of a ``b`` argument for reports.
 
    Returns
    -------
    str
        * ``"fn"`` for a function;
        * ``repr(b)`` for a scalar, for example ``"4.0"``;
        * for an array, its length and first and last values, for example
          ``"array(len=100000, 2->20)"``.
    """
    if callable(b):
        return "fn"
    if np.isscalar(b):
        return repr(b)
    arr = np.asarray(b)
    return f"array(len={len(arr)}, {arr[0]:.3g}->{arr[-1]:.3g})"


def _run_attempt_jit(graph: CSRGraph, A: float, B: float, b, max_iters: int,
                      seed: int, init_fn=None) -> AttemptResult:
    """Run one attempt with :func:`solver_jit.petford_welsh_jit`.
 
    A new ``np.random.default_rng(seed)`` is created for the attempt. All
    other arguments are passed straight to the solver. The returned
    ``elapsed`` is the solver's own timing.
    """
    rng = np.random.default_rng(seed)
    _, best_value, best_step, elapsed = petford_welsh_jit(
        graph, A=A, B=B, b=b, max_iters=max_iters, rng=rng, init_fn=init_fn
    )
    return AttemptResult(seed=seed, best_value=int(best_value), best_step=int(best_step), elapsed=elapsed)

def _load_graph(graph_or_path: CSRGraph | str, name: str | None = None) -> CSRGraph:
    """Return a graph unchanged, or load it from an edge-list file.
 
    Parameters
    ----------
    graph_or_path : CSRGraph or str
        A graph, returned as is, or a path to an edge-list file.
    name : str, optional
        Name for a loaded graph. By default, the file name with
        ``_stable_set_edge_list.txt`` removed. A file without that suffix
        keeps its full name, including ``.txt``.
 
    Returns
    -------
    CSRGraph
    """
    if isinstance(graph_or_path, CSRGraph):
        return graph_or_path
    path = graph_or_path
    inferred_name = name or os.path.basename(path).replace("_stable_set_edge_list.txt", "")
    return CSRGraph.from_edge_list_file(path, name=inferred_name)


def run_multi_start(
    graph: CSRGraph | str,
    *,
    A: float = 1.0,
    B: float = 2.0,
    b: float = 4.0,
    max_iters: int = 100000,
    num_attempts: int = 10,
    seed: int = 0,
    known_alpha: int | None = None,
    init_fn = None,
) -> ExperimentResult:
    """Run the solver several times on one instance with consecutive seeds.
 
    Attempt ``i`` uses seed ``seed + i``. All attempts share the same graph
    and parameters.
 
    Parameters
    ----------
    graph : CSRGraph or str
        A graph, or a path to an edge-list file. A file is loaded once
        before the attempts start.
    A : float, default 1.0
        Reward for each vertex in the set.
    B : float, array-like, or callable, default 2.0
        Conflict penalty.
    b : float, array-like, or callable, default 4.0
        Petford-Welsh base.
    max_iters : int, default 100000
        Iterations per attempt.
    num_attempts : int, default 10
        Number of attempts.
    seed : int, default 0
        Seed of the first attempt.
    known_alpha : int, optional
        Known stability number, used to compute ``gap``.
    init_fn : InitFn, optional
        Starting-state generator, for example
        :func:`init_fns.random_order_init_jit`. If omitted, attempts start
        from the empty set.
 
    Returns
    -------
    ExperimentResult
        Per-attempt results and summary statistics.
    """
    g = _load_graph(graph)

    result = ExperimentResult(
        instance=g.name, n=g.n, m=g.m,
        A=A, B_repr=_repr(B), b_repr=_repr(b), max_iters=max_iters, num_attempts=num_attempts, known_alpha=known_alpha,
    )

    t0 = perf_counter()
    for i in range(num_attempts):
        result.attempts.append(_run_attempt_jit(g, A, B, b, max_iters, seed+i, init_fn))
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
    """Run :func:`run_multi_start` on every ``.txt`` file in a directory.
 
    Files are processed in sorted file-name order.
 
    Parameters
    ----------
    instances_dir : str
        Directory containing the edge-list files. It is not searched
        recursively.
    known_alpha : dict of str to int, optional
        Known stability numbers, keyed by instance name (for example
        ``"c5"``, not the file name). Instances missing from the dict get
        ``known_alpha=None``.
    name_filter : callable, optional
        ``name_filter(filename) -> bool``, called with the full file name
        including the suffix. Only files for which it returns ``True`` are
        run.
    console_output : bool, default True
        Print a one-line summary after each instance. The gap is shown as
        ``?`` when unknown.
    **run_kwargs
        Passed on to :func:`run_multi_start`, for example ``solver``,
        ``A``, ``B``, ``b``, ``max_iters``, ``num_attempts``, ``seed``,
        ``target`` or ``init_fn``.
 
    Returns
    -------
    list of ExperimentResult
        One result per instance, in processing order.
    """
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
            print(f"{result.instance:35s} n={result.n:5d} m={result.m:7d} "
                  f"best={result.best:4d} mean={result.mean:7.2f} std={result.std:5.2f} "
                  f"gap={gap_str} time={result.total_time:6.1f}s")
    return results

def save_results_csv(results: list, path: str) -> None:
    """Write one summary row per instance to a CSV file.
 
    Columns are those of :meth:`ExperimentResult.to_row`.
 
    Parameters
    ----------
    results : list of ExperimentResult
        Results to write. If the list is empty, nothing is written and no
        file is created.
    path : str
        Output file. It is overwritten if it exists, and missing parent
        directories are created.

    """
    if not results:
        return
    rows = [r.to_row() for r in results]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def save_attempts_csv(results: list, path: str) -> None:
    """Write one row per attempt, for all instances, to a CSV file.
 
    Columns: ``instance, solver, seed, best_value, best_step, elapsed,
    known_alpha``. Unknown ``known_alpha`` is written as ``""``.
 
    Parameters
    ----------
    results : list of ExperimentResult
        Results to write. If there are no attempts, nothing is written and
        no file is created.
    path : str
        Output file. It is overwritten if it exists, and missing parent
        directories are created.
    """
    rows = []
    for r in results:
        for a in r.attempts:
            rows.append({
                "instance": r.instance, "seed": a.seed,
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