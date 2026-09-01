"""
Sweeps the two hyperparameters of a cooling schedule for petford_welsh /
petford_welsh_jit -- b_start (schedule's value at it=0) and b_end (value at
it=max_iters-1) -- for both a geometric and a linear schedule shape, AND
across a grid of max_iters budgets, rendering one row of heatmaps per
budget (one column per schedule shape, shared color scale across the whole
figure).

Why max_iters is swept too, not fixed: b_start/b_end are stretched over
exactly max_iters steps, so the "best" (b_start, b_end) found by sweeping
at a single fixed max_iters is confounded with that budget. With few
iterations there's no time to explore, so the sweep will just reward
"end cold" (large b_end) regardless of whether annealing shape actually
helped -- being cold the whole time wins by default when you can't afford
to wander. With many iterations there's room for a hot start to actually
pay off. A single-budget heatmap can't tell those two stories apart; it
just tells you what happens to be true for that one budget. Faceting by
max_iters is what makes that distinction visible: watch whether the
argmin's (b_start, b_end) moves as you go down the rows, and whether it's
consistently near the diagonal (b_start ~ b_end, i.e. no real annealing
benefit) or consistently displaced from it (real benefit to a hot start
given enough iterations to use it).

b relates to temperature via T = 1/ln(b) (see solver.py/solver_jit.py), so
LOW b = HIGH T (hot, exploratory) and HIGH b = LOW T (cold, exploitative).
The two schedule shapes:

    geometric: b(it) = b_start * (b_end / b_start) ** (it / (max_iters - 1))
               -- np.geomspace(b_start, b_end, max_iters)
    linear:    b(it) = b_start + (b_end - b_start) * it / (max_iters - 1)
               -- np.linspace(b_start, b_end, max_iters)

Both are built as plain float64 arrays (not callables) and passed straight
to run_multi_start -- the array branch in petford_welsh_jit is vectorized
(`temps = 1/np.log(b)`), while the callable branch evaluates b(it) in a
Python list comprehension once per iteration, which would dominate the
sweep's wall time for no reason.

The diagonal b_start == b_end (marked in red on every heatmap) is the
fixed-temperature baseline (what the project already runs by default,
b=4.0) and is identical for both shapes at every budget -- it's what "no
annealing benefit at this budget" looks like.

b must be > 1 for ln(b) > 0 (finite positive temperature) -- the default
grid respects that.

WARNING: this is len(b_grid)^2 * len(schedules) * len(max_iters_grid) *
num_attempts runs. Defaults (9x9 grid, 2 schedules, 4 budgets, 10
attempts) = 6480 runs -- cut down --b-grid / --max-iters-grid /
--num-attempts for a quick look first.

Requires: numpy, matplotlib
    pip install numpy matplotlib
"""
import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scripts import *

KNOWN_ALPHA = {
    "C125.9": 34, "MANN_a9": 16, "brock200-4": 17,
    "brock800_1": 23, "brock800_2": 24, "brock800_3": 25, "brock800_4": 26,
    "c-fat200-1": 12, "c-fat200-2": 24, "c-fat200-5": 58,
    "c-fat500-1": 14, "c-fat500-2": 26, "c-fat500-5": 64,
    "dsjc125.5": 10, "dsjc125.9": 34,
    "evil-N120-p98-chv12x10": 20, "evil-N120-p98-myc5x24": 48,
    "evil-N121-p98-myc11x11": 22, "evil-N125-p98-s3m25x5": 20,
    "hamming6_2": 32, "hamming6_4": 4,
    "johnson16_2_4": 8, "johnson8_2_4": 4, "johnson8_4_4": 14,
    "keller4": 11, "p-hat500-1": 9,
    "p_hat1500_1": 12, "p_hat1500_2": 65, "p_hat1500_3": 94,
    "paley101": 5, "paley61": 5, "paley73": 5, "paley89": 5, "paley97": 6,
    "san200-0-7-1": 30, "san200-0-7-2": 18, "sanr200-0-7": 18,
}

DEFAULT_B_GRID = [1.5, 2, 3, 5, 8, 12, 20, 35, 50]
DEFAULT_MAX_ITERS_GRID = [20000, 100000, 500000, 2000000]
A = 1.0
B = 10.0
SEED = 0

SCHEDULE_BUILDERS = {
    "geometric": lambda b_start, b_end, max_iters: np.geomspace(b_start, b_end, num=max_iters),
    "linear":    lambda b_start, b_end, max_iters: np.linspace(b_start, b_end, num=max_iters),
}


def run_sweep(graph, b_grid, max_iters, num_attempts, seed, solver, build_schedule):
    """
    avg_best[i, j], avg_time[i, j] for b_start=b_grid[i], b_end=b_grid[j],
    at this one max_iters.
    """
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
    parser.add_argument("--instance", default="C125.9")
    parser.add_argument("--b-grid", type=float, nargs="+", default=DEFAULT_B_GRID)
    parser.add_argument("--max-iters-grid", type=int, nargs="+", default=DEFAULT_MAX_ITERS_GRID)
    parser.add_argument("--num-attempts", type=int, default=10)
    parser.add_argument("--solver", choices=["python", "jit"], default="jit")
    parser.add_argument("--schedules", choices=list(SCHEDULE_BUILDERS), nargs="+",
                         default=["geometric", "linear"])
    parser.add_argument("--no-annotate", action="store_true", help="skip per-cell value labels")
    args = parser.parse_args()

    if any(b <= 1.0 for b in args.b_grid):
        raise ValueError(f"all b values must be > 1 (got {args.b_grid}); b<=1 makes ln(b)<=0, i.e. non-finite/negative temperature")

    base_dir = os.getcwd()
    name = args.instance
    target = KNOWN_ALPHA.get(name)
    path = os.path.join(base_dir, "instances", "stable_set", f"{name}_stable_set_edge_list.txt")
    graph = CSRGraph.from_edge_list_file(path, name=name)

    if args.solver == "jit":
        petford_welsh_jit(graph, A=A, B=B, b=4.0, max_iters=10, rng=np.random.default_rng(1))

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
            print(f"{'b_start':>8s} {'b_end':>8s} {'avg_best':>9s} {'avg_time':>9s}")
            for i, b_start in enumerate(args.b_grid):
                for j, b_end in enumerate(args.b_grid):
                    print(f"{b_start:8.2f} {b_end:8.2f} {avg_best[i, j]:9.2f} {avg_time[i, j]:9.4f}")

            metric = (target - avg_best) if target is not None else avg_best
            metric_by_max_iters[max_iters][schedule_name] = metric

    metric_label = "avg optimality gap (known alpha - best found)" if target is not None else "avg best found"
    cmap = "viridis_r" if target is not None else "viridis"

    # Shared color scale across the WHOLE figure (all budgets, all schedules)
    # -- deliberately, so you can also see budgets closing the gap toward 0
    # as max_iters grows, not just which schedule wins within one budget.
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