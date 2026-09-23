"""Sweep the Petford-Welsh (A, B, b) parameter grid across instances with a
known alpha, and rank each (B/A ratio, base) combination by its average
relative optimality gap.

For each instance that has a KNOWN_ALPHA entry, runs the solver at every
combination of --ratio-grid (B/A) and --base-grid (b), --num-attempts
attempts per combination. For every run, relative_gap =
(known_alpha - best) / known_alpha. Every (instance, ratio, base) row is
written to results/param_sweep_results.csv. The average relative gap
across instances for each (ratio, base) pair is printed as a ranking
(lowest gap first) and drawn as an imshow heatmap to
results/param_sweep_gap_heatmap.png.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from benchmark_common import *
from scripts import *

BASES = [3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20, 25, 30, 40, 50, 75, 100]
B_RATIOS = [1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]


def run_grid(names, ratios, bases, A, max_iters, num_attempts, seed, base_dir):
    """Run every instance at every (ratio, base) combination.

    Returns
    -------
    rows : list of dict
        One row per (instance, ratio, base), for the CSV.
    avg_gap : np.ndarray, shape (len(ratios), len(bases))
        Average relative_gap across instances, for each (ratio, base) pair.
    """
    rows = []
    gap_sum = np.zeros((len(ratios), len(bases)))
    gap_count = np.zeros((len(ratios), len(bases)))

    for name in tqdm(names):
        graph = load_instance(name, base_dir)
        known_alpha = KNOWN_ALPHA[name]

        for i, ratio in enumerate(ratios):
            B = A * ratio
            for j, base in enumerate(bases):
                result = run_multi_start(
                    graph, A=A, B=B, b=base,
                    max_iters=max_iters, num_attempts=num_attempts, seed=seed,
                    known_alpha=known_alpha, show_progress=False,
                )
                relative_gap = (known_alpha - result.best) / known_alpha
                rows.append({
                    "instance": name, "n": graph.n, "m": graph.m,
                    "A": A, "B": B, "ratio": ratio, "base": base,
                    "best": result.best, "mean": result.mean, "std": result.std,
                    "known_alpha": known_alpha, "gap": result.gap,
                    "relative_gap": relative_gap,
                })
                gap_sum[i, j] += relative_gap
                gap_count[i, j] += 1

    avg_gap = gap_sum / gap_count
    return rows, avg_gap


def plot_heatmap(ratios, bases, avg_gap, out_path, cmap):
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(bases) + 2), max(5, 0.5 * len(ratios) + 2)))
    im = ax.imshow(avg_gap, origin="lower", cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(bases)))
    ax.set_xticklabels([f"{b:g}" for b in bases], rotation=45, ha="right")
    ax.set_yticks(range(len(ratios)))
    ax.set_yticklabels([f"{r:g}" for r in ratios])
    ax.set_xlabel("base b")
    ax.set_ylabel("B/A ratio")
    ax.set_title("average relative gap  (known_alpha - best) / known_alpha")

    vmin, vmax = float(np.nanmin(avg_gap)), float(np.nanmax(avg_gap))
    mid = (vmin + vmax) / 2
    for i in range(len(ratios)):
        for j in range(len(bases)):
            val = avg_gap[i, j]
            color = "white" if val > mid else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=7)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("average relative gap")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    add_instances_arg(parser)
    parser.add_argument("--ratio-grid", type=float, nargs="+", default=B_RATIOS)
    parser.add_argument("--base-grid", type=float, nargs="+", default=BASES)
    parser.add_argument("--A", type=float, default=DEFAULT_A)
    parser.add_argument("--max-iters", type=int, default=DEFAULT_MAX_ITERS)
    parser.add_argument("--num-attempts", type=int, default=DEFAULT_NUM_ATTEMPTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cmap", default="viridis_r",
                         help="matplotlib colormap for the heatmap (lower gap = better)")
    parser.add_argument("--output", default=None,
                         help="heatmap PNG path; default results/param_sweep_gap_heatmap.png")
    args = parser.parse_args()

    base_dir = os.getcwd()

    if args.instances:
        missing = [n for n in args.instances if n not in KNOWN_ALPHA]
        if missing:
            raise ValueError(f"no known_alpha for: {missing}; relative gap needs a known alpha")
        names = args.instances
    else:
        names = [n for n in all_instance_names(base_dir) if n in KNOWN_ALPHA]

    total_runs = len(names) * len(args.ratio_grid) * len(args.base_grid) * args.num_attempts
    print(f"{len(names)} instances with known_alpha, {len(args.ratio_grid)}x{len(args.base_grid)} grid, "
          f"{args.num_attempts} attempts each -> {total_runs} total PW runs")

    rows, avg_gap = run_grid(
        names, args.ratio_grid, args.base_grid, args.A,
        args.max_iters, args.num_attempts, args.seed, base_dir,
    )

    csv_path = results_path("param_sweep_results.csv", base_dir)
    write_csv(rows, csv_path)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    print("\n=== Average relative gap ranking (lower is better) ===")
    ranked = sorted(
        ((args.ratio_grid[i], args.base_grid[j], avg_gap[i, j])
         for i in range(len(args.ratio_grid)) for j in range(len(args.base_grid))),
        key=lambda t: t[2],
    )
    for ratio, base, gap in ranked:
        print(f"  ratio={ratio:>6g} base={base:>5g}: avg_relative_gap={gap:.4f}")

    best_ratio, best_base, best_gap = ranked[0]
    print(f"\nRecommended default: A={args.A:g}, B={args.A * best_ratio:g}, base b={best_base:g} "
          f"(avg_relative_gap={best_gap:.4f})")

    out_path = args.output or results_path("param_sweep_gap_heatmap.png", base_dir)
    out_path = plot_heatmap(args.ratio_grid, args.base_grid, avg_gap, out_path, args.cmap)
    print(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    main()