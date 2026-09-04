import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from benchmark_common import *
from scripts import *

DEFAULT_B_GRID = [1.5, 2, 3, 5, 8, 12, 20, 35, 50]
DEFAULT_MAX_ITERS_GRID = [20000, 100000, 500000, 2000000]
A, B, B_SCHEDULE, SEED = DEFAULT_A, DEFAULT_B, DEFAULT_BASE, DEFAULT_SEED

SCHEDULE_BUILDERS = {
    "geometric": lambda b_start, b_end, max_iters: np.geomspace(b_start, b_end, num=max_iters),
    "linear":    lambda b_start, b_end, max_iters: np.linspace(b_start, b_end, num=max_iters),
}


def run_sweep(graph, b_grid, max_iters, num_attempts, seed, solver, build_schedule):
    ng = len(b_grid)
    avg_best = np.zeros((ng, ng))
    avg_time = np.zeros((ng, ng))
    for i, b_start in enumerate(b_grid):
        for j, b_end in enumerate(b_grid):
            b_schedule = build_schedule(b_start, b_end, max_iters)
            result = run_multi_start(
                graph, solver=solver, A=A, B=B, b=b_schedule,
                max_iters=max_iters, num_attempts=num_attempts, seed=seed,
            )
            vals = [a.best_value for a in result.attempts]
            times = [a.elapsed for a in result.attempts]
            avg_best[i, j] = float(np.mean(vals))
            avg_time[i, j] = float(np.mean(times))
    return avg_best, avg_time


def main():
    parser = argparse.ArgumentParser()
    add_instance_arg(parser)
    parser.add_argument("--b-grid", type=float, nargs="+", default=DEFAULT_B_GRID)
    parser.add_argument("--max-iters-grid", type=int, nargs="+", default=DEFAULT_MAX_ITERS_GRID)
    parser.add_argument("--num-attempts", type=int, default=10)
    add_solver_arg(parser)
    parser.add_argument("--schedules", choices=list(SCHEDULE_BUILDERS), nargs="+",
                         default=["geometric", "linear"])
    parser.add_argument("--no-annotate", action="store_true", help="skip per-cell value labels")
    args = parser.parse_args()

    if any(b <= 1.0 for b in args.b_grid):
        raise ValueError(f"all b values must be > 1 (got {args.b_grid}); b<=1 makes ln(b)<=0, i.e. non-finite/negative temperature")

    base_dir = os.getcwd()
    name = args.instance
    target = KNOWN_ALPHA.get(name)
    graph = load_instance(name, base_dir)

    if args.solver == "jit":
        jit_warmup(graph, A=A, B=B, b=B_SCHEDULE, seed=1)

    total_runs = len(args.b_grid) ** 2 * len(args.schedules) * len(args.max_iters_grid) * args.num_attempts
    print(f"--- {name} (n={graph.n}, m={graph.m}) known alpha={target} ---")
    print(f"sweeping b_start x b_end over {len(args.b_grid)}x{len(args.b_grid)} grid, "
          f"max_iters_grid={args.max_iters_grid}, num_attempts={args.num_attempts}, solver={args.solver}, "
          f"schedules={args.schedules}")
    print(f"-> {total_runs} total PW runs")

    # metric_by_max_iters[max_iters][schedule] -> (ng, ng) array
    metric_by_max_iters = {}
    for max_iters in args.max_iters_grid:
        metric_by_max_iters[max_iters] = {}
        for schedule_name in args.schedules:
            t0 = time.perf_counter()
            avg_best, avg_time = run_sweep(
                graph, args.b_grid, max_iters, args.num_attempts, SEED, args.solver,
                SCHEDULE_BUILDERS[schedule_name],
            )
            elapsed = time.perf_counter() - t0
            print(f"\n[max_iters={max_iters} schedule={schedule_name}] sweep took {elapsed:.1f}s")

            metric = (target - avg_best) if target is not None else avg_best
            metric_by_max_iters[max_iters][schedule_name] = metric

    metric_label = "avg optimality gap (known alpha - best found)" if target is not None else "avg best found"
    cmap = "viridis_r" if target is not None else "viridis"

    all_vals = np.concatenate([
        m.ravel() for per_schedule in metric_by_max_iters.values() for m in per_schedule.values()
    ])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    mid = (vmin + vmax) / 2

    n = len(args.b_grid)
    n_rows = len(args.max_iters_grid)
    n_cols = len(args.schedules)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 6.5 * n_rows), squeeze=False)

    im = None
    for row, max_iters in enumerate(args.max_iters_grid):
        for col, schedule_name in enumerate(args.schedules):
            ax = axes[row][col]
            metric = metric_by_max_iters[max_iters][schedule_name]
            im = ax.imshow(metric, origin="lower", cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

            ax.set_xticks(range(n))
            ax.set_xticklabels([f"{b:g}" for b in args.b_grid], rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(n))
            ax.set_yticklabels([f"{b:g}" for b in args.b_grid], fontsize=8)
            if row == n_rows - 1:
                ax.set_xlabel("b_end (colder finish ->)")
            if col == 0:
                ax.set_ylabel(f"max_iters={max_iters}\nb_start (hotter start ->)")
            ax.set_title(schedule_name if row == 0 else "", fontsize=11)

            for i in range(n):
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.5))

            if not args.no_annotate:
                for i in range(n):
                    for j in range(n):
                        color = "white" if metric[i, j] < mid else "black"
                        ax.text(j, i, f"{metric[i, j]:.1f}", ha="center", va="center", color=color, fontsize=6)

    fig.suptitle(f"{name}: cooling schedule sweep across budgets ({metric_label})")
    cbar = fig.colorbar(im, ax=axes, shrink=0.7)
    cbar.set_label(metric_label)

    schedules_tag = "-".join(args.schedules)
    out_path = os.path.join(base_dir, "results", f"schedule_sweep_by_budget_{name}_{schedules_tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()