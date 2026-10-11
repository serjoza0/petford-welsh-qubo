import os
import csv
import argparse
from datetime import datetime

from benchmark_common import *
from scripts import run_multi_start
from scripts import run_instance as ri

DEFAULT_TIERS = [("extra_small", 10), ("small", 100), ("medium", 1000),
                 ("large", 10000), ("extra_large", 100000),
                 ("huge", 1000000), ("giant", 10000000)]
DEFAULT_TIME_LIMIT = 300.0

FIELDS = [
    "instance", "tier", "factor", "n", "m", "max_iters", "num_attempts", "seed",
    "best", "mean", "std", "elapsed_sec", "upper_bound", "reached_optimum", "timestamp",
]


def parse_tiers(specs):
    if not specs:
        return list(DEFAULT_TIERS)
    tiers = []
    for spec in specs:
        name, sep, value = spec.partition("=")
        if not sep or not name:
            raise SystemExit(f"bad --tiers entry {spec!r}; expected name=factor, e.g. small=100")
        try:
            factor = float(value)
        except ValueError:
            raise SystemExit(f"bad --tiers entry {spec!r}; factor must be a number")
        if factor <= 0:
            raise SystemExit(f"bad --tiers entry {spec!r}; factor must be positive")
        if name in [t for t, _ in tiers]:
            raise SystemExit(f"tier name {name!r} given twice")
        tiers.append((name, factor))
    return tiers


def row_key(instance, tier, factor, num_attempts, seed):
    return (str(instance), str(tier), f"{factor:g}", str(num_attempts), str(seed))


def load_done(path):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if any(row.get(k) is None for k in FIELDS):
                continue
            done[row_key(row["instance"], row["tier"], float(row["factor"]),
                         row["num_attempts"], row["seed"])] = row
    return done


def append_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    needs_newline = False
    if not needs_header:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_newline = f.read(1) != b"\n"
    with open(path, "a", newline="") as f:
        if needs_newline:
            f.write("\n")
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def instance_state(name, done, tiers, args):
    spent = 0.0
    prev = None
    for i, (tier, factor) in enumerate(tiers):
        row = done.get(row_key(name, tier, factor, args.num_attempts, args.seed))
        if row is None:
            predicted = prev[1] * factor / prev[0] if prev is not None else 0.0
            if spent + predicted > args.time_limit:
                return "time_limit", i, spent, predicted
            return "pending", i, spent, predicted
        elapsed = float(row["elapsed_sec"])
        spent += elapsed
        if str(row["reached_optimum"]) == "1":
            return "solved", i, spent, 0.0
        prev = (tiers[i][1], elapsed)
    return "exhausted", len(tiers), spent, 0.0


def process_instance(name, label, tiers, done, upper_bound, args, base_dir, output, warmed):
    graph = None
    ran = False
    while True:
        status, i, spent, predicted = instance_state(name, done, tiers, args)
        if status != "pending":
            break
        tier, factor = tiers[i]
        if graph is None:
            graph = load_instance(name, base_dir)
            if not warmed[0]:
                jit_warmup(graph)
                warmed[0] = True
        max_iters = max(1, round(graph.n * factor))
        ub = upper_bound.get(name)

        if predicted >= 10:
            print(f"{label} {name:28s} {tier:11s} n={graph.n:6d} max_iters={max_iters:11d} "
                  f"running, estimated {predicted:.0f}s ...", flush=True)

        result = run_multi_start(
            graph, A=ri.DEFAULT_A, B=ri.DEFAULT_B, b=ri.DEFAULT_BASE,
            max_iters=max_iters, num_attempts=args.num_attempts, seed=args.seed,
            known_alpha=ub, show_progress=False,
        )
        reached = ub is not None and result.best >= ub

        row = {
            "instance": name, "tier": tier, "factor": f"{factor:g}",
            "n": graph.n, "m": graph.m, "max_iters": max_iters,
            "num_attempts": args.num_attempts, "seed": args.seed,
            "best": result.best, "mean": f"{result.mean:.4f}", "std": f"{result.std:.4f}",
            "elapsed_sec": f"{result.total_time:.3f}",
            "upper_bound": ub if ub is not None else "",
            "reached_optimum": 1 if reached else 0,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        append_row(output, row)
        done[row_key(name, tier, factor, args.num_attempts, args.seed)] = row
        ran = True

        if ub is None:
            verdict = "no upper bound known"
        elif reached:
            verdict = "OPTIMUM reached"
        else:
            verdict = f"gap={ub - result.best}"
        print(f"{label} {name:28s} {tier:11s} n={graph.n:6d} max_iters={max_iters:11d} "
              f"best={result.best:5d} ub={ub if ub is not None else '?':>5}  {verdict}  "
              f"({result.total_time:.1f}s)", flush=True)

    if ran and status == "time_limit":
        print(f"{label} {name}: stopped, {spent:.0f}s used and next tier '{tiers[i][0]}' is "
              f"estimated at {predicted:.0f}s (limit {args.time_limit:g}s)", flush=True)
    return status


def print_summary(names, done, tiers, upper_bound, args):
    print(f"\n=== summary (tiers: {', '.join(f'{t}={f:g}' for t, f in tiers)}; "
          f"attempts={args.num_attempts}, seed={args.seed}, time limit={args.time_limit:g}s) ===")
    solved_in = {t: 0 for t, _ in tiers}
    unsolved, timed_out, no_ub, pending = [], [], [], []
    for name in names:
        status, i, spent, _ = instance_state(name, done, tiers, args)
        best = max((int(r["best"]) for t, f in tiers
                    if (r := done.get(row_key(name, t, f, args.num_attempts, args.seed))) is not None),
                   default=None)
        if status == "solved":
            solved_in[tiers[i][0]] += 1
        elif status == "pending":
            pending.append(name)
        elif name not in upper_bound:
            no_ub.append((name, best, status))
        elif status == "time_limit":
            timed_out.append((name, best, upper_bound[name], tiers[i - 1][0] if i else "-"))
        else:
            unsolved.append((name, best, upper_bound[name]))

    print("optimum reached at: " + ", ".join(f"{t} {c}" for t, c in solved_in.items()))
    print(f"not reached after all tiers: {len(unsolved)}  |  stopped by time limit: {len(timed_out)}"
          f"  |  no upper bound known: {len(no_ub)}  |  not finished yet: {len(pending)}")
    if unsolved:
        print("\nnot reached after all tiers (best / upper_bound):")
        for name, best, ub in unsolved:
            print(f"  {name:30s} {best} / {ub}  (gap {ub - best})")
    if timed_out:
        print("\nstopped by time limit (best / upper_bound, last tier run):")
        for name, best, ub, last in timed_out:
            print(f"  {name:30s} {best} / {ub}  (gap {ub - best}, {last})")
    if no_ub:
        print("\nno upper bound known (best found):")
        for name, best, status in no_ub:
            print(f"  {name:30s} {best}" + ("  (stopped by time limit)" if status == "time_limit" else ""))
    if pending:
        print(f"\nnot finished yet ({len(pending)}): rerun the same command to continue")


def main():
    parser = argparse.ArgumentParser()
    add_instances_arg(parser)
    parser.add_argument("--tiers", nargs="+", default=None, metavar="NAME=FACTOR",
                         help="tiers in order, each as name=factor; per-attempt max_iters = factor * n. "
                              "Default: extra_small=10 small=100 medium=1000 large=10000 extra_large=100000 "
                              "huge=1000000 giant=10000000")
    parser.add_argument("--num-attempts", type=int, default=DEFAULT_NUM_ATTEMPTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT,
                         help="seconds one instance may use in total over its tiers; an instance is "
                              "stopped before a tier that would push it past this (default 300)")
    parser.add_argument("--output", default=None,
                         help="results CSV, appended to and used to resume; "
                              "default results/budget_tiers_results.csv")
    args = parser.parse_args()

    base_dir = os.getcwd()
    tiers = parse_tiers(args.tiers)
    output = args.output or results_path("budget_tiers_results.csv", base_dir)

    available = all_instance_names(base_dir)
    names = args.instances or available
    unknown = [n for n in names if n not in available]
    if unknown:
        raise SystemExit(f"unknown instances: {unknown}")

    upper_bound = load_upper_bounds(base_dir)
    done = load_done(output)
    finished = sum(1 for n in names if instance_state(n, done, tiers, args)[0] != "pending")
    print(f"{len(names)} instances, {len(upper_bound)} with a known upper bound; "
          f"{len(done)} finished runs found in {output}; "
          f"{finished} instances already finished, {len(names) - finished} to go", flush=True)

    warmed = [False]
    try:
        for idx, name in enumerate(names, 1):
            process_instance(name, f"[{idx}/{len(names)}]", tiers, done, upper_bound,
                             args, base_dir, output, warmed)
    except (KeyboardInterrupt, SystemError) as e:
        if isinstance(e, SystemError) and not isinstance(e.__cause__, KeyboardInterrupt):
            raise
        print("\ninterrupted -- finished runs are saved; rerun the same command to continue")

    print_summary(names, done, tiers, upper_bound, args)


if __name__ == "__main__":
    main()
