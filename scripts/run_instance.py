import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from scripts import *

DEFAULT_A = 1.0
DEFAULT_B = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0])
DEFAULT_BASE = np.array([1.5, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0])
DEFAULT_SEED = 0
DEFAULT_NUM_ATTEMPTS = 1000
DEFAULT_TIME_LIMIT = 3.0
CALIB_ITERS = 1000000
CALIB_ATTEMPTS = 100

def calibrate_rate(graph, seed):
    run_multi_start(graph, A=DEFAULT_A, B=DEFAULT_B[0], b=DEFAULT_BASE[0],
                    max_iters=10, num_attempts=1, seed=seed, show_progress=False)
    result = run_multi_start(graph, A=DEFAULT_A, B=DEFAULT_B, b=DEFAULT_BASE,
                             max_iters=CALIB_ITERS, num_attempts=CALIB_ATTEMPTS, seed=seed, show_progress=False)
    elapsed = result.total_time
    return (CALIB_ITERS * CALIB_ATTEMPTS) / elapsed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run Petford-Welsh on a single instance file, by path. "
            "Give exactly two of --max-iters/--time-per-attempt (one slot, pick a unit), "
            "--num-attempts, and --time-limit (total, across every attempt) -- the third is derived."
        ),
    )
    parser.add_argument("instance_path", help="path to a *_stable_set_edge_list.txt file")
    parser.add_argument("--A", type=float, default=DEFAULT_A)
    parser.add_argument("--B", type=float, default=DEFAULT_B, help="B/A ratio penalty")
    parser.add_argument("--base", type=float, default=DEFAULT_BASE, help="Petford-Welsh base b")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-iters", type=int, default=None, help="iterations per attempt")
    parser.add_argument("--time-per-attempt", type=float, default=None,
                         help="seconds per attempt; alternative unit for --max-iters, converted via a "
                              "quick calibration run on this graph")
    parser.add_argument("--num-attempts", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=None,
                         help="total wall-clock seconds across all attempts combined")
    parser.add_argument("--known-alpha", type=int, default=None)
    parser.add_argument("--print-attempts", action="store_true",
                         help="print one line per attempt (seed, B, b, best, step, elapsed)")
    parser.add_argument("--save-csv", default=None,
                         help="write one row per attempt to this CSV path")
    args = parser.parse_args()

    if args.max_iters is not None and args.time_per_attempt is not None:
        raise SystemExit("pass only one of --max-iters or --time-per-attempt -- they're two units for "
                          "the same thing (effort per attempt)")

    has_effort = args.max_iters is not None or args.time_per_attempt is not None
    has_num_attempts = args.num_attempts is not None
    has_time_limit = args.time_limit is not None
    n_given = sum([has_effort, has_num_attempts, has_time_limit])

    if n_given == 0:
        args.num_attempts = DEFAULT_NUM_ATTEMPTS
        args.time_limit = DEFAULT_TIME_LIMIT
        has_num_attempts = has_time_limit = True
    elif n_given == 1:
        if has_effort or has_time_limit:
            args.num_attempts = DEFAULT_NUM_ATTEMPTS
            has_num_attempts = True
        else:
            args.time_limit = DEFAULT_TIME_LIMIT
            has_time_limit = True
    elif n_given == 3:
        raise SystemExit(
            "give exactly two of: --max-iters (or --time-per-attempt), --num-attempts, --time-limit, "
            "not all three -- the third is derived, not an independent input."
        )

    if not os.path.isfile(args.instance_path):
        raise SystemExit(f"instance file not found: {args.instance_path}")
    name = os.path.basename(args.instance_path).replace("_stable_set_edge_list.txt", "")
    graph = CSRGraph.from_edge_list_file(args.instance_path, name=name)

    rate = None
    def get_rate():
        nonlocal rate
        if rate is None:
            rate = calibrate_rate(graph, args.seed)
            print(f"calibration: ~{rate:.0f} iters/sec on this graph")
        return rate

    if has_effort and has_num_attempts:
        max_iters = args.max_iters if args.max_iters is not None else max(1, round(get_rate() * args.time_per_attempt))
        num_attempts = args.num_attempts

    elif has_effort and has_time_limit:
        if args.time_per_attempt is not None:
            if args.time_per_attempt > args.time_limit:
                raise SystemExit(f"--time-per-attempt={args.time_per_attempt} exceeds --time-limit={args.time_limit}")
            time_per_attempt = args.time_per_attempt
            max_iters = max(1, round(get_rate() * time_per_attempt))
        else:
            max_iters = args.max_iters
            time_per_attempt = max_iters / get_rate()
        num_attempts = max(1, int(args.time_limit // time_per_attempt))

    else:  # has_num_attempts and has_time_limit
        num_attempts = args.num_attempts
        time_per_attempt = args.time_limit / num_attempts
        max_iters = max(1, round(get_rate() * time_per_attempt))

    predicted_total = None
    if rate is not None:
        predicted_total = num_attempts * max_iters / rate

    print(f"{name}: n={graph.n}  m={graph.m}   A={args.A}  B={args.B}  base={args.base}")
    print(f"resolved: max_iters={max_iters}  num_attempts={num_attempts}"
          + (f"  (~{predicted_total:.1f}s predicted)" if predicted_total is not None else ""))

    result = run_multi_start(
        graph, A=args.A, B=args.B, b=args.base,
        max_iters=max_iters, num_attempts=num_attempts, seed=args.seed,
        known_alpha=args.known_alpha,
    )

    print(f"\nran {num_attempts} attempts x {max_iters} iterations in {result.total_time:.2f}s "
          f"(actual, vs. predicted above)")
    print(f"best={result.best}  mean={result.mean:.2f}  std={result.std:.2f}")
    if args.known_alpha is not None:
        print(f"known alpha: {args.known_alpha}  gap: {result.gap}")

    if args.print_attempts:
        print(f"\n{'seed':>6s} {'B':>8s} {'b':>8s} {'best':>6s} {'step':>10s} {'elapsed':>10s}")
        for a in result.attempts:
            print(f"{a.seed:6d} {a.B:8g} {a.b:8g} {a.best_value:6d} {a.best_step:10d} {a.elapsed:10.4f}")

    if args.save_csv:
        save_attempts_csv([result], args.save_csv)
        print(f"\nwrote {len(result.attempts)} attempt rows to {args.save_csv}")


if __name__ == "__main__":
    main()