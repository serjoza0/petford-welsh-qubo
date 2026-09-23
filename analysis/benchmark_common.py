import os
import sys
import glob
import csv
from time import perf_counter
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scripts.graph import CSRGraph, read_edge_list, create_networkx_graph
from scripts.solver_jit import petford_welsh_jit

DEFAULT_A = 1
DEFAULT_B = 2
DEFAULT_BASE = 8
DEFAULT_SEED = 0
DEFAULT_MAX_ITERS = 1000000
DEFAULT_NUM_ATTEMPTS = 100

ATTEMPT_BUCKETS = [(130, 500), (250, 500), (600, 200), (float("inf"), 100)]
QUICK_ATTEMPT_BUCKETS = [(130, 5), (250, 5), (600, 3), (float("inf"), 2)]


# INSTANCE_SUBDIRS = ["dimacs"]
INSTANCE_SUBDIRS = ["dimacs", "bhoslib", "PACE", "quantum", "EVIL", "coding_theory"]

def instances_dir(base_dir=None):
    return os.path.join(base_dir or os.getcwd(), "instances")

def _instance_file_index(base_dir=None):
    base = instances_dir(base_dir)
    index = {}
    for sub in INSTANCE_SUBDIRS:
        sub_dir = os.path.join(base, sub)
        if not os.path.isdir(sub_dir):
            continue
        for path in sorted(glob.glob(os.path.join(sub_dir, "*_stable_set_edge_list.txt"))):
            name = os.path.basename(path).replace("_stable_set_edge_list.txt", "")
            if name not in index:
                index[name] = path
    return index

def instance_path(name, base_dir=None):
    index = _instance_file_index(base_dir)
    if name not in index:
        raise FileNotFoundError(f"no instance named {name!r} found under {instances_dir(base_dir)}")
    return index[name]

def all_instance_names(base_dir=None):
    return sorted(_instance_file_index(base_dir))

def load_instance(name, base_dir=None):
    return CSRGraph.from_edge_list_file(instance_path(name, base_dir), name=name)

def load_instance_nx(name, base_dir=None):
    n, edges = read_edge_list(instance_path(name, base_dir))
    return create_networkx_graph(n, edges)

def iter_graphs(names=None, base_dir=None, max_n=None, verbose=True):
    index = _instance_file_index(base_dir)
    names = names or sorted(index)
    total = len(names)
    for idx, name in enumerate(names):
        path = index.get(name)
        if path is None:
            if verbose:
                print(f"[{idx+1}/{total}] {name}: instance file not found, skipping")
            continue
        graph = CSRGraph.from_edge_list_file(path, name=name)
        if max_n is not None and graph.n > max_n:
            if verbose:
                print(f"[{idx+1}/{total}] {name}: n={graph.n} > max-n={max_n}, skipping")
            continue
        yield idx, total, name, graph

def attempts_for(n, buckets=ATTEMPT_BUCKETS):
    for max_n, num_attempts in buckets:
        if n <= max_n:
            return num_attempts
    raise RuntimeError("Check value of n")

def jit_warmup(graph, A=DEFAULT_A, B=DEFAULT_B, b=DEFAULT_BASE, seed=1):
    petford_welsh_jit(graph, A=A, B=B, b=b, max_iters=10, rng=np.random.default_rng(seed))

def make_annealing_schedule(max_iters, b_lo, b_hi, kind="geometric"):
    if kind == "geometric":
        return np.geomspace(b_lo, b_hi, num=max_iters)
    if kind == "linear":
        return np.linspace(b_lo, b_hi, num=max_iters)
    raise ValueError(f"unknown schedule kind: {kind!r}")

def run_sa_attempts(nx_graph, num_attempts, beta, sampler):
    from qpu_comparison import calculate_best_solution, eliminate_and_recalculate
    out = []
    for _ in range(num_attempts):
        t0 = perf_counter()
        solutions = calculate_best_solution(
            nx_graph, sampler, beta=beta, num_of_runs=1,
            no_output_file=True, console_output=False,
        )
        sample_set = solutions[0]["sample_set"]
        rec = eliminate_and_recalculate(nx_graph, sample_set, beta, sampler=sampler, num_of_runs=1, console_output=False)
        elapsed = perf_counter() - t0
        best = len(rec["best_recalculated_solution"]["recalculated_solution_nodes"])
        out.append((best, elapsed))
    return out

def write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def results_path(filename, base_dir=None):
    return os.path.join(base_dir or os.getcwd(), "results", filename)

def add_instance_arg(parser, default="C125.9"):
    parser.add_argument("--instance", default=default)
    return parser

def add_instances_arg(parser):
    parser.add_argument("--instances", nargs="+", default=None,
                         help="specific instance names; default is every instance found under "
                              "instances/ (across all subfolders)")
    return parser

def add_pw_args(parser, default_max_iters=DEFAULT_MAX_ITERS, default_num_attempts=DEFAULT_NUM_ATTEMPTS, default_seed=DEFAULT_SEED):
    parser.add_argument("--max-iters", type=int, default=default_max_iters)
    parser.add_argument("--num-attempts", type=int, default=default_num_attempts)
    parser.add_argument("--seed", type=int, default=default_seed)
    parser.add_argument("--A", type=float, default=DEFAULT_A)
    parser.add_argument("--B", type=float, default=DEFAULT_B)
    parser.add_argument("--base", type=float, default=DEFAULT_BASE)
    return parser

def add_skip_sa_arg(parser):
    parser.add_argument("--skip-sa", action="store_true")
    return parser