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

DEFAULT_B_GRID = [1.5, 2, 3, 5, 8, 12, 20, 35, 50, 100, 200]
DEFAULT_MAX_ITERS = 100000
DEFAULT_NUM_ATTEMPTS = 5
A = 1.0
B = 1.5
SEED = 0
NORMALIZE_BY = "alpha"  # "alpha" (default, see docstring) or "n" (graph size)

SCHEDULE_BUILDERS = {
    "geometric": lambda b_start, b_end, max_iters: np.geomspace(b_start, b_end, num=max_iters),
    "linear":    lambda b_start, b_end, max_iters: np.linspace(b_start, b_end, num=max_iters),
}


def run_sweep_rel_gap(graph, target, b_grid, max_iters, num_attempts, seed, solver, build_schedule):
    """rel_gap[i, j] = (target - avg_best) / normalizer for b_start=b_grid[i], b_end=b_grid[j]."""
    ng = len(b_grid)
    rel_gap = np.zeros((ng, ng))
    normalizer = target if NORMALIZE_BY == "alpha" else graph.n
    for i, b_start in enumerate(b_grid):
        for j, b_end in enumerate(b_grid):
            b_schedule = build_schedule(b_start, b_end, max_iters)
            result = run_multi_start(
                graph, solver=solver, A=A, B=B, b=b_schedule,
                max_iters=max_iters, num_attempts=num_attempts, seed=seed,
            )
            avg_best = float(np.mean([a.best_value for a in result.attempts]))
            rel_gap[i, j] = (target - avg_best) / normalizer
    return rel_gap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", nargs="+", default=list(KNOWN_ALPHA))
    parser.add_argument("--b-grid", type=float, nargs="+", default=DEFAULT_B_GRID)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--num-attempts", type=int, default=DEFAULT_NUM_ATTEMPTS)
    parser.add_argument("--solver", choices=["python", "jit"], default="jit")
    parser.add_argument("--schedules", choices=list(SCHEDULE_BUILDERS), nargs="+",
                         default=["geometric", "linear"])
    parser.add_argument("--no-annotate", action="store_true", help="skip per-cell value labels")
    parser.add_argument("--yes", action="store_true", help="skip the run-count confirmation prompt")
    args = parser.parse_args()

    if any(b <= 1.0 for b in args.b_grid):
        raise ValueError(f"all b values must be > 1 (got {args.b_grid}); b<=1 makes ln(b)<=0, i.e. non-finite/negative temperature")

    unknown = [name for name in args.instances if name not in KNOWN_ALPHA]
    if unknown:
        raise ValueError(f"no known alpha for: {unknown} -- rel_gap needs a target; remove them or add to KNOWN_ALPHA")

    n_cells = len(args.b_grid) ** 2
    total_runs = n_cells * len(args.schedules) * len(args.instances) * args.num_attempts
    print(f"instances={len(args.instances)}  b_grid={len(args.b_grid)}x{len(args.b_grid)}  "
          f"schedules={args.schedules}  num_attempts={args.num_attempts}  max_iters={args.max_iters}")
    print(f"-> {total_runs} total PW runs")
    if not args.yes:
        resp = input("proceed? [y/N] ").strip().lower()
        if resp != "y":
            print("aborted.")
            return

    base_dir = os.getcwd()

    if args.solver == "jit":
        # Warm up numba once on the first instance before timing anything.
        warm_path = os.path.join(base_dir, "instances", "stable_set", f"{args.instances[0]}_stable_set_edge_list.txt")
        warm_graph = CSRGraph.from_edge_list_file(warm_path, name=args.instances[0])
        petford_welsh_jit(warm_graph, A=A, B=B, b=4.0, max_iters=10, rng=np.random.default_rng(1))

    # rel_gap_sums[schedule] accumulates the sum across instances; divide by
    # count at the end. Accumulating instead of storing every instance's
    # full grid keeps memory flat regardless of how many instances you run.
    rel_gap_sums = {s: np.zeros((len(args.b_grid), len(args.b_grid))) for s in args.schedules}
    n_ok = 0
    t_start = time.perf_counter()

    for idx, name in enumerate(args.instances):
        target = KNOWN_ALPHA[name]
        path = os.path.join(base_dir, "instances", "stable_set", f"{name}_stable_set_edge_list.txt")
        try:
            graph = CSRGraph.from_edge_list_file(path, name=name)
        except FileNotFoundError:
            print(f"[{idx+1}/{len(args.instances)}] {name}: instance file not found, skipping")
            continue

        print(f"[{idx+1}/{len(args.instances)}] {name} (n={graph.n}, m={graph.m}, alpha={target})", end="", flush=True)
        t0 = time.perf_counter()
        for schedule_name in args.schedules:
            rel_gap = run_sweep_rel_gap(
                graph, target, args.b_grid, args.max_iters, args.num_attempts, SEED, args.solver,
                SCHEDULE_BUILDERS[schedule_name],
            )
            rel_gap_sums[schedule_name] += rel_gap
        n_ok += 1
        print(f" -- {time.perf_counter() - t0:.1f}s")

    total_elapsed = time.perf_counter() - t_start
    print(f"\ndone: {n_ok}/{len(args.instances)} instances in {total_elapsed:.1f}s")
    if n_ok == 0:
        print("no instances succeeded, nothing to plot.")
        return

    avg_rel_gap = {s: rel_gap_sums[s] / n_ok for s in args.schedules}

    print(f"\nmean rel_gap over {n_ok} instances ({'alpha' if NORMALIZE_BY == 'alpha' else 'n'}-normalized):")
    for schedule_name in args.schedules:
        print(f"\n[{schedule_name}]")
        print(f"{'b_start':>8s} {'b_end':>8s} {'mean_rel_gap':>13s}")
        for i, b_start in enumerate(args.b_grid):
            for j, b_end in enumerate(args.b_grid):
                print(f"{b_start:8.2f} {b_end:8.2f} {avg_rel_gap[schedule_name][i, j]:13.4f}")

    all_vals = np.concatenate([m.ravel() for m in avg_rel_gap.values()])
    vmin, vmax = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
    mid = (vmin + vmax) / 2

    n = len(args.b_grid)
    ns = len(args.schedules)
    fig, axes = plt.subplots(1, ns, figsize=(7.5 * ns, 7.0), squeeze=False)
    axes = axes[0]

    im = None
    for ax, schedule_name in zip(axes, args.schedules):
        metric = avg_rel_gap[schedule_name]
        im = ax.imshow(metric, origin="lower", cmap="viridis_r", aspect="equal", vmin=vmin, vmax=vmax)

        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{b:g}" for b in args.b_grid], rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"{b:g}" for b in args.b_grid])
        ax.set_xlabel("b_end (value at it=max_iters-1; higher = colder finish)")
        ax.set_ylabel("b_start (value at it=0; lower = hotter start)")
        ax.set_title(schedule_name)

        for i in range(n):
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2))

        if not args.no_annotate:
            for i in range(n):
                for j in range(n):
                    color = "white" if metric[i, j] < mid else "black"
                    ax.text(j, i, f"{metric[i, j]:.3f}", ha="center", va="center", color=color, fontsize=7)

    norm_label = "alpha" if NORMALIZE_BY == "alpha" else "n"
    fig.suptitle(f"mean rel. gap over {n_ok} instances -- (alpha - best)/{norm_label}, averaged")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label(f"mean rel. gap ((alpha - best)/{norm_label})")

    schedules_tag = "-".join(args.schedules)
    out_path = os.path.join(base_dir, "results", f"schedule_sweep_all_instances_{schedules_tag}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")


if __name__ == "__main__":
    main()