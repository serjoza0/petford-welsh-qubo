import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from benchmark_common import *
from scripts import *

from dwave.samplers import SimulatedAnnealingSampler

A, B, B_SCHEDULE, SEED = DEFAULT_A, DEFAULT_B, DEFAULT_BASE, DEFAULT_SEED
SA_BETA = 0.5

def constant_schedule(max_iters, b0=4.0):
    return np.full(max_iters, b0, dtype=np.float64)

def linear_schedule(max_iters, b_lo, b_hi):
    return np.linspace(b_lo, b_hi, max_iters)

def geometric_schedule(max_iters, b_lo, b_hi):
    t = np.linspace(0.0, 1.0, max_iters)
    return b_lo * (b_hi / b_lo) ** t

def oscillating_schedule(max_iters, b_mid, b_amp, n_cycles):
    it = np.arange(max_iters)
    return b_mid + b_amp * np.sin(2 * np.pi * n_cycles * it / max_iters)


SCHEDULES = {
    "constant (b=4)":               lambda mi: constant_schedule(mi, 4.0),
    "linear up (2->50)":            lambda mi: linear_schedule(mi, 2.0, 50.0),
    "linear down (50->2)":          lambda mi: linear_schedule(mi, 50.0, 2.0),
    "geometric mild (2->20)":       lambda mi: geometric_schedule(mi, 2.0, 20.0),
    "geometric steep (1.05->200)":  lambda mi: geometric_schedule(mi, 1.05, 200.0),
    "oscillating (2<->18, 5 cyc)":  lambda mi: oscillating_schedule(mi, 10.0, 8.0, 5),
}
MAX_ITERS = [5000, 20000, 100000, 500000, 2000000]


def main():
    parser = argparse.ArgumentParser()
    add_instance_arg(parser)
    parser.add_argument("--num-attempts", type=int, default=20)
    add_solver_arg(parser)
    add_skip_sa_arg(parser)
    args = parser.parse_args()

    base_dir = os.getcwd()
    name = args.instance
    target = KNOWN_ALPHA.get(name)
    graph = load_instance(name, base_dir)

    if args.solver == "jit":
        petford_welsh_jit(graph, A=A, B=B, b=4.0, max_iters=10, rng=np.random.default_rng(0))  # warm up numba

    print(f"--- {name} (n={graph.n}, m={graph.m}) known alpha={target} ---")
    print(f"{'schedule':32s} {'max_iters':>9s} {'avg_time':>10s} {'avg_best':>9s} {'avg_gap':>8s}")


    group_stats = {}
    for sched_name, sched_fn in SCHEDULES.items():
        rows = []
        for max_iters in MAX_ITERS:
            b = sched_fn(max_iters)
            result = run_multi_start(
                graph, solver=args.solver, A=A, B=B, b=b,
                max_iters=max_iters, num_attempts=args.num_attempts, seed=SEED,
            )
            avg_time = float(np.mean([a.elapsed for a in result.attempts]))
            avg_best = result.mean
            avg_gap = (target - avg_best) if target is not None else None
            rows.append((max_iters, avg_time, avg_best, avg_gap))
            gap_str = f"{avg_gap:8.2f}" if avg_gap is not None else "     n/a"
            print(f"{sched_name:32s} {max_iters:9d} {avg_time:10.4f} {avg_best:9.2f} {gap_str}")
        group_stats[sched_name] = rows

    sa_avg = None
    if not args.skip_sa:
        sampler = SimulatedAnnealingSampler()
        nx_graph = load_instance_nx(name, base_dir)
        sa_points = run_sa_attempts(nx_graph, args.num_attempts, SA_BETA, sampler)
        sa_avg_time = float(np.mean([t for _, t in sa_points]))
        sa_avg_best = float(np.mean([v for v, _ in sa_points]))
        sa_avg_gap = (target - sa_avg_best) if target is not None else None
        sa_avg = (sa_avg_time, sa_avg_gap if sa_avg_gap is not None else sa_avg_best)
        gap_str = f"{sa_avg_gap:8.2f}" if sa_avg_gap is not None else "     n/a"
        print(f"{'SA (reference)':32s} {'-':>9s} {sa_avg_time:10.4f} {sa_avg_best:9.2f} {gap_str}")

    # --- plot: averages only ---
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    sched_names = list(SCHEDULES.keys())
    color_for = {n_: cmap(i % 10) for i, n_ in enumerate(sched_names)}

    for sched_name, rows in group_stats.items():
        rows_sorted = sorted(rows, key=lambda r: r[0])
        xs = [r[1] for r in rows_sorted]
        ys = [(r[3] if r[3] is not None else r[2]) for r in rows_sorted]
        ax.plot(xs, ys, color=color_for[sched_name], marker="D", markersize=9,
                markeredgecolor="black", linewidth=1.5, alpha=0.9, zorder=4)


    if sa_avg is not None:
        ax.scatter([sa_avg[0]], [sa_avg[1]], color="crimson", marker="^", s=130,
                   edgecolor="black", linewidth=1.2, zorder=5)

    ax.set_xlabel("avg time (s)")
    ax.set_ylabel("avg optimality gap (known alpha - avg best found)" if target is not None else "avg best found")
    ax.set_title(f"{name}: schedule averages across max_iters budgets" + (" + SA reference" if sa_avg else ""))
    ax.set_xscale("log")

    handles = [Line2D([0], [0], marker="D", color=color_for[n_], markerfacecolor=color_for[n_],
                       markeredgecolor="black", markersize=8, label=n_) for n_ in sched_names]
    if sa_avg is not None:
        handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="crimson",
                               markeredgecolor="black", markersize=9, label="SA (reference)"))
    ax.legend(handles=handles, loc="best", fontsize=7)

    out_path = os.path.join(base_dir, "results", f"schedule_budget_sweep_{name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    n_points = sum(len(rows) for rows in group_stats.values())

if __name__ == "__main__":
    main()