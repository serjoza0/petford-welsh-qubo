import os
import sys

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from benchmark_common import *
from scripts import *

from dwave.samplers import SimulatedAnnealingSampler


DEFAULT_MAX_ITERS_LIST = [5000, 20000, 100000, 500000, 2000000, 10000000, 50000000]
SA_BETA = 0.5
BOX_WIDTH_FRAC = 0.15

INIT_STRATEGIES = {
    "empty":      None,
    "greedy":     greedy_min_degree_init,
    "greedy_jit": greedy_min_degree_init_jit,
}
COLOR_FOR = {"empty": "tab:blue", "greedy": "tab:orange", "greedy_jit": "tab:green"}



def run_pw_sweep(graph, max_iters_list, num_attempts, seed, solver, init_fn):
    points = []
    for max_iters in max_iters_list:
        result = run_multi_start(
            graph, solver=solver, A=DEFAULT_A, B=DEFAULT_B, b=DEFAULT_BASE,
            max_iters=max_iters, num_attempts=num_attempts, seed=seed,
            init_fn=init_fn,
        )
        for i, a in enumerate(result.attempts):
            points.append((max_iters, a.best_value, result.total_time))
    return points

def main():
    parser = argparse.ArgumentParser()
    add_instance_arg(parser)
    parser.add_argument("--num-attempts", type=int, default=DEFAULT_NUM_ATTEMPTS)
    add_solver_arg(parser)
    add_skip_sa_arg(parser)
    args = parser.parse_args()

    base_dir = os.getcwd()
    name = args.instance
    target = KNOWN_ALPHA.get(name)
    graph = load_instance(name, base_dir)

    if args.solver == "jit":
        jit_warmup(graph, A=DEFAULT_A, B=DEFAULT_B, b=DEFAULT_BASE, seed=DEFAULT_SEED)

    greedy_min_degree_init_jit(graph, 1, np.random.default_rng(DEFAULT_SEED))

    print(f"--- {name} (n={graph.n}, m={graph.m}) known alpha={target} ---")
    print(f"{'init':>7s} {'max_iters':>10s} {'avg_time':>10s} {'avg_best':>9s} {'avg_gap':>8s}")

    all_points = {}
    group_stats = {}
    for init_label, init_fn in INIT_STRATEGIES.items():
        pw_points = run_pw_sweep(graph, DEFAULT_MAX_ITERS_LIST, args.num_attempts, DEFAULT_SEED, args.solver, init_fn)
        all_points[init_label] = pw_points

        rows = []
        for max_iters in DEFAULT_MAX_ITERS_LIST:
            vals = [(v, t) for (mi, v, t) in pw_points if mi == max_iters]
            best_vals = [v for v, t in vals]
            times_ = [t for v, t in vals]
            avg_time, avg_best = float(np.mean(times_)), float(np.mean(best_vals))
            avg_gap = (target - avg_best) if target is not None else None
            rows.append((max_iters, avg_time, avg_best, avg_gap))
            gap_str = f"{avg_gap:8.2f}" if avg_gap is not None else "     n/a"
            print(f"{init_label:>7s} {max_iters:10d} {avg_time:10.4f} {avg_best:9.2f} {gap_str}")
        group_stats[init_label] = rows

    sa_points = []
    if not args.skip_sa:
        sampler = SimulatedAnnealingSampler()
        nx_graph = load_instance_nx(name, base_dir)
        sa_points = run_sa_attempts(nx_graph, args.num_attempts, SA_BETA, sampler)
        sa_times = [t for _, t in sa_points]
        sa_bests = [v for v, _ in sa_points]
        sa_avg_gap = (target - float(np.mean(sa_bests))) if target is not None else float("nan")
        print(f"\nSA reference: avg_time={np.mean(sa_times):.4f}s avg_best={np.mean(sa_bests):.2f} avg_gap={sa_avg_gap:.2f}")

    fig, ax = plt.subplots(figsize=(9, 6.5))

    def y_of(v):
        return (target - v) if target is not None else v

    for init_label in INIT_STRATEGIES:
        pw_points = all_points[init_label]
        rows = group_stats[init_label]
        box_data, positions, widths = [], [], []
        for max_iters, avg_time, avg_best, avg_gap in rows:
            ys = [y_of(v) for (mi, v, t) in pw_points if mi == max_iters]
            box_data.append(ys)
            positions.append(avg_time)
            widths.append(avg_time * BOX_WIDTH_FRAC)

        bp = ax.boxplot(
            box_data, positions=positions, widths=widths, patch_artist=True,
            manage_ticks=False, showmeans=True, meanline=True, zorder=4,
            medianprops=dict(color="black", linewidth=1.3),
            meanprops=dict(color="black", linewidth=1.3, linestyle="--"),
            whiskerprops=dict(color="black"), capprops=dict(color="black"),
            flierprops=dict(marker="o", markersize=4, markeredgecolor="black", alpha=0.5),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(COLOR_FOR[init_label])
            patch.set_alpha(0.65)
            patch.set_edgecolor("black")

    if sa_points:
        for v, t in sa_points:
            ax.scatter(t, y_of(v), color="crimson", marker="^", alpha=0.6, s=30, zorder=5)


    ax.set_xlabel("time (s) -- includes construction time for greedy starts")
    ax.set_ylabel("optimality gap (known alpha - best found)" if target is not None else "best found")
    ax.set_title(f"{name}: init strategies across max_iters budgets" + (" + SA reference" if sa_points else ""))
    ax.set_xscale("log")

    handles = [
        Patch(facecolor=COLOR_FOR[label], edgecolor="black", alpha=0.65, label=f"{label} start")
        for label in INIT_STRATEGIES
    ]
    if sa_points:
        handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="crimson", markersize=8, label="SA (reference)"))
    ax.legend(handles=handles, loc="best", fontsize=8)

    out_path = os.path.join(base_dir, "results", f"greedy_start_compare_{name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    n_points = sum(len(pts) for pts in all_points.values())
    print(f"\nSaved plot to {out_path}")

if __name__ == "__main__":
    main()