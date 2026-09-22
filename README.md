# Maximum Stable Set via Petford-Welsh Simulated Annealing
 
Solves the Maximum Stable Set Problem (MSSP) with a Petford-Welsh-style
simulated annealing solver. Vertices are 0/1 variables; the energy
 
```
E(x) = -A * sum(x_v) + B * sum(x_u * x_v for edges {u, v})
```
 
rewards including a vertex (`A`) and penalizes edges with both endpoints
included (`B`). Each iteration proposes flipping one vertex and accepts the
flip with probability `1 / (1 + b ** (B*s - A))`, where `s` is the number of
currently-included neighbors and `b > 1` is the Petford-Welsh base
(`temperature = 1 / ln(b)`).
 
## Requirements
 
- Python >= 3.10
- Dependencies listed in `requirements.txt`
Install with:
 
```
pip install -r requirements.txt
```
 
## Project layout
 
- `scripts/` — the core package: graph representation and I/O (`graph.py`),
  the solver (`solver_jit.py`), starting-state generators (`init_fns.py`),
  multi-attempt experiment running and CSV output (`experiment.py`), and a
  command-line entry point (`run_instance.py`).
- `analysis/` — benchmark and sweep scripts built on top of `scripts/`. Some
  predate the current per-attempt `B`/`b` array API and may need updating
  before they run.
- `instances/` — MSSP instance files as edge lists (`stable_set/` and the
  BHOSLIB family under `bhoslib/`).
- `results/` — CSVs and plots produced by the analysis scripts.
- `report/` — LaTeX write-up.
## Instance file format
 
Plain text edge-list files, 1-indexed:
 
```
n m
u1 v1
u2 v2
...
```
 
An optional third column (edge weight) is accepted but ignored. Vertices are
converted to 0-indexed internally when the file is loaded.
 
## Running a single instance
 
```
python scripts/run_instance.py instances/stable_set/C125.9_stable_set_edge_list.txt
```
 
Give exactly two of `--max-iters` (or `--time-per-attempt`), `--num-attempts`,
and `--time-limit`; the third is derived from a short calibration run on the
instance. `--B` and `--base` accept either a single value or, by default, a
grid of values that attempts cycle through as the cross product `(B, b)`.
`--print-attempts` prints one line per attempt, and `--save-csv PATH` writes
the same per-attempt data to a CSV file. Run with `--help` for the full list
of options.