# LCCP — reproducibility material

This repository is supplementary material for the paper *"A Numerically-safe
Branch-Price-and-Cut Algorithm for the Length-Constrained Cycle Partition
Problem"*, submitted to the *European Journal of Operational Research* (EJOR).

It contains the per-instance result data and the scripts that turn it into every
number reported in the computational study (Section 7), so that the tables and
headline figures can be regenerated from the raw data.

## Quick start

```bash
pip install -r requirements.txt
python3 compute_paper_numbers.py
```

`compute_paper_numbers.py` is self-contained (only `pandas` + `numpy`) and
prints, with the paper's exact conventions, the numbers behind:

- **Table 1** (overall performance, dual side) and the speedup figures;
- the **headline** (51 solved, largest 76 nodes, 14 previously-unsolved closed);
- the **ablation**, **cutting-plane**, **numerical-safety**, **dual-stabilization**
  and **branching** tables/paragraphs.

## Data: `results/`

One CSV per solver configuration, one row per benchmark instance (84 instances,
14–100 nodes, from the standard benchmark set of Hoppmann-Baum et al. 2022).
Columns include `primal_bound`, `dual_bound`, `dual_bound_at_root`, `final_gap`,
`total_time`, `number_of_nodes`, `cg_iterations_root`, `cg_iterations_total`.

| CSV | Paper label | Description |
|---|---|---|
| `compact.csv` | `bnc-sec` | Gurobi branch-and-cut (SEC), MCV warm start |
| `compact_bks.csv` | `bnc-full` | Gurobi branch-and-cut (SEC), best-known-solution warm start |
| `27_full_nobestsol.csv` | `bnp-mcv` | branch-and-price, MCV warm start |
| `27_full.csv` | `bnp-full` | branch-and-price, best-known-solution warm start |
| `27_full_nobidir/_nopar/_nosymbr/_noearly/_nogreedy/_trieq.csv` | ablation | one feature toggled off (or trieq on) |
| `27_clique/_srow/_srow_clique/_srow_pricing.csv` | cutting planes | added cut families |
| `27_full_unsafe.csv` | `bnp-unsafe` | floating-point (non-safe) bounds |
| `27_full_stabilized.csv` | — | dual stabilization |
| `27_full_rf_nobestsol.csv` | — | Ryan-Foster branching (no warm start) |

All runs: two-hour time limit, exclusive node access, on the hardware described
in Section 7 (2× Intel Xeon Gold 6338). Solving times are wall-clock.

## Scripts

- **`compute_paper_numbers.py`** — primary entry point (see above).
- **`generate_paper_data.py`** + **`analyze.py`** — the original analysis code.
  Reproduces the same tables plus the **per-instance supplementary table**
  (`analyze.generate_supplementary_tables`) and the **greedy-pricing /
  ng-relaxation** statistics (success rate, neighbourhood expansions), which are
  parsed from the raw solver logs. Needs `tqdm` + `matplotlib`.
- **`extract_info.py`** (branch-and-price logs), **`extract_info_compact.py`**
  (Gurobi/SEC logs), **`aggregator.py`**, **`process_logs.sh`**,
  **`process_logs_compact.sh`** — the pipeline that turns raw `.sol` solver logs
  into the CSVs in `results/`. The raw `.sol` logs are large and are not shipped
  here; the CSVs are the reproducible unit. To rebuild a CSV from logs:

  ```bash
  bash process_logs.sh <log-dir>          # branch-and-price configs
  bash process_logs_compact.sh <log-dir>  # Gurobi SEC (compact) configs
  ```

## Notes

- **`bnc-sec` / `bnc-full`** were re-run with Gurobi 13.0.1; the original SEC
  runner from the prior work was unavailable. `bnc-full` (best-known warm start)
  reproduces the previously published baseline (39 solved, up to 52 nodes).
- The **heuristic configurations** (`heurbnp`, `heurbnp-rf`) and the **primal-side
  columns** of Table 1 are carried over from the earlier experimental setup and
  are not regenerated here (see the Table 1 caption footnote).
- `compute_paper_numbers.py` measures the Table 1 root gap against the optimum
  $z^*$ over the instances where `bnp-full`'s root LP converges, while the
  ablation table's `Gap` column reports the average root gap per configuration —
  the two conventions are documented inline in the script.
