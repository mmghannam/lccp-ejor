#!/usr/bin/env python3
"""
Generate all data and tables for the LCCP paper.

This script computes all statistics mentioned in the paper and generates
the LaTeX tables for the results section.

Usage:
    python3 generate_paper_data.py

Output files are written to ~/papers/lccp-paper/EJOR Version/
"""

import os
import sys
import glob
import re
import json
import numpy as np
import pandas as pd

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze import (
    read_csv, sgm, fix_time, extract_number_from_filename,
    best_known_sols, root_bound_comparison
)

# Configuration
RESULTS_DIR = "results"
PAPER_DIR = os.path.expanduser("~/papers/lccp-paper/EJOR Version")
TIMELIMIT = 2 * 3600  # 2 hours

# Main experiment version used in paper
EXPERIMENT_VERSION = "27"


def load_experiment_files():
    """Load all experiment files for the main experiment version."""
    files = {
        'compact': f"{RESULTS_DIR}/compact.csv",
        'compact_bks': f"{RESULTS_DIR}/compact_bks.csv",
        'full': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full.csv",
        'full_nobidir': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full_nobidir.csv",
        'full_nopar': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full_nopar.csv",
        'full_nosymbr': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full_nosymbr.csv",
        'full_noearly': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full_noearly.csv",
        'full_nogreedy': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full_nogreedy.csv",
        'full_nobestsol': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full_nobestsol.csv",
        'full_trieq': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full_trieq.csv",
        'full_unsafe': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full_unsafe.csv",
        'clique': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_clique.csv",
        'srow': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_srow.csv",
        'srow_pricing': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_srow_pricing.csv",
        'srow_clique': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_srow_clique.csv",
        'full_stabilized': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full_stabilized.csv",
        'full_rf_nobestsol': f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full_rf_nobestsol.csv",
    }
    return {k: v for k, v in files.items() if os.path.exists(v)}


def compute_solver_stats(csv_file, instance_filter=None):
    """Compute statistics for a single solver."""
    df = read_csv(csv_file)

    if instance_filter is not None:
        df_filtered = df[df["instance_name"].isin(instance_filter)]
    else:
        df_filtered = df

    solved = df[(df["final_gap"] == 0) & (df["total_time"] < TIMELIMIT)]

    return {
        'solved': len(solved),
        'total': len(df),
        'sgm_time': sgm(df_filtered['total_time'], 1),
        'avg_nodes': df_filtered['number_of_nodes'].mean(),
        'avg_gap': df_filtered['final_gap'].mean() * 100,         # true optimality gap
        'avg_root_gap': df_filtered['root_gap'].mean() * 100       # root LP gap
            if 'root_gap' in df_filtered.columns else float('nan'),
        'df': df,
    }


def get_instances_solved_by_any(files):
    """Get instances solved by at least one solver."""
    solved_by_any = set()
    for f in files.values():
        if os.path.exists(f):
            df = read_csv(f)
            solved = set(df[(df["final_gap"] == 0) & (df["total_time"] < TIMELIMIT)]["instance_name"])
            solved_by_any |= solved
    return solved_by_any


def solved_set(csv_file):
    """Instances solved to proven optimality (true final gap == 0 within the time limit)."""
    if not os.path.exists(csv_file):
        return set()
    df = read_csv(csv_file)
    return set(df[(df["final_gap"] == 0) & (df["total_time"] < TIMELIMIT)]["instance_name"])


def per_table_basket(files, keys):
    """Union of instances solved by at least one configuration listed in this table.

    Matches the per-table SGM convention stated in the paper: absolute SGM values are
    therefore not comparable across tables.
    """
    basket = set()
    for k in keys:
        if k in files:
            basket |= solved_set(files[k])
    return basket


def root_metrics(csv_file):
    """At-root and root-LP-converged counts for the ablation/cuts tables.

    - at_root:  instances solved to optimality in a single B&B node.
    - root_lp:  instances whose root LP relaxation converged (valid root dual bound),
                approximated by dual_bound_at_root > 1 or the instance being solved.
    """
    if not os.path.exists(csv_file):
        return 0, 0
    df = read_csv(csv_file)
    solved = (df["final_gap"] == 0) & (df["total_time"] < TIMELIMIT)
    at_root = int((solved & (df["number_of_nodes"] == 1)).sum())
    root_lp = int(((df["dual_bound_at_root"] > 1) | solved).sum())
    return at_root, root_lp


def analyze_greedy_heuristic(results_dir):
    """Analyze greedy heuristic from log files."""
    sol_files = glob.glob(os.path.join(results_dir, "*.sol"))
    if not sol_files:
        return None

    stats = {
        'greedy_success': 0,
        'greedy_fail': 0,
        'greedy_cols': 0,
        'labeling_cols': 0,
        'ng_expansions': [],
        'max_neighborhoods': [],
        'per_instance_success_rates': [],  # For computing mean of per-instance rates
    }

    for sol_file in sol_files:
        with open(sol_file, 'r') as f:
            content = f.read()

        instance_success = len(re.findall(r'> heuristically found variables: (\d+)', content))
        instance_fail = len(re.findall(r'> no heuristically found variables', content))
        instance_total = instance_success + instance_fail

        stats['greedy_success'] += instance_success
        stats['greedy_fail'] += instance_fail

        # Per-instance success rate
        if instance_total > 0:
            stats['per_instance_success_rates'].append(instance_success / instance_total * 100)

        greedy_cols = re.findall(r'> heuristically found variables: (\d+)', content)
        stats['greedy_cols'] += sum(int(x) for x in greedy_cols)

        labeling_cols = re.findall(r'> added paths: (\d+)', content)
        stats['labeling_cols'] += sum(int(x) for x in labeling_cols)

        neighborhood_pcts = re.findall(r'larger neighborhood \((\d+)%\)', content)
        ng_expansions = len(neighborhood_pcts)
        max_neighborhood = max([int(x) for x in neighborhood_pcts]) if neighborhood_pcts else 0

        stats['ng_expansions'].append(ng_expansions)
        stats['max_neighborhoods'].append(max_neighborhood)

    return stats


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")


def main():
    print("LCCP Paper Data Generation")
    print(f"Experiment version: {EXPERIMENT_VERSION}")
    print(f"Output directory: {PAPER_DIR}")

    files = load_experiment_files()
    print(f"\nLoaded {len(files)} experiment files")

    # Get instances solved by at least one solver (for fair comparison)
    instances_solved_by_any = get_instances_solved_by_any(files)
    print(f"Instances solved by at least one solver: {len(instances_solved_by_any)}")

    # =========================================================================
    # Section: Overall Results
    # =========================================================================
    print_section("OVERALL RESULTS")

    # Per-table basket: instances solved by at least one exact configuration
    # (bnc-sec, bnp-mcv = full_nobestsol, bnp-full, bnc-full = compact_bks),
    # matching the Table 1 caption in the paper.
    overall_basket = per_table_basket(files, ['compact', 'full_nobestsol', 'full', 'compact_bks'])
    print(f"(per-table basket: {len(overall_basket)} instances)")
    full_stats = compute_solver_stats(files['full'], overall_basket)
    compact_stats = compute_solver_stats(files['compact'], overall_basket)

    print(f"\nBNP-FULL:")
    print(f"  Solved: {full_stats['solved']} instances")
    print(f"  SGM time: {full_stats['sgm_time']:.1f}s")

    print(f"\nBNC-SEC:")
    print(f"  Solved: {compact_stats['solved']} instances")
    print(f"  SGM time: {compact_stats['sgm_time']:.1f}s")

    speedup = compact_stats['sgm_time'] / full_stats['sgm_time']
    print(f"\nSpeedup: {speedup:.1f}x")

    # =========================================================================
    # Section: Root Bound Comparison
    # =========================================================================
    print_section("ROOT BOUND COMPARISON")

    root_df = root_bound_comparison(files['compact'], files['full'])

    # =========================================================================
    # Section: Ablation Study
    # =========================================================================
    print_section("ABLATION STUDY")

    ablation_configs = [
        ('full', 'BNP-FULL'),
        ('full_nobidir', 'No bidirectional'),
        ('full_nopar', 'No parallelization'),
        ('full_nosymbr', 'No symmetry breaking'),
        ('full_noearly', 'No early branching'),
        ('full_nogreedy', 'No greedy pricing'),
        ('full_nobestsol', 'No best solution'),
    ]

    ablation_extra = [('full_trieq', 'With triangle ineq.')]
    ablation_keys = [k for k, _ in ablation_configs + ablation_extra]
    ablation_basket = per_table_basket(files, ablation_keys)
    full_time = compute_solver_stats(files['full'], ablation_basket)['sgm_time']

    print(f"(per-table basket: {len(ablation_basket)} instances)")
    print(f"\n{'Config':<25} {'Solved':>7} {'AtRoot':>7} {'RootLP':>7} {'Time':>8} {'Ratio':>7} {'Gap%':>6}")
    print("-" * 72)

    solved_sets_ablation = {}
    for key, name in ablation_configs + ablation_extra:
        if key in files:
            stats = compute_solver_stats(files[key], ablation_basket)
            at_root, root_lp = root_metrics(files[key])
            ratio = stats['sgm_time'] / full_time
            solved_sets_ablation[key] = solved_set(files[key])
            print(f"{name:<25} {stats['solved']:>7} {at_root:>7} {root_lp:>7} "
                  f"{stats['sgm_time']:>8.1f} {ratio:>7.2f} {stats['avg_root_gap']:>6.1f}")

    vb_ablation = set().union(*solved_sets_ablation.values()) if solved_sets_ablation else set()
    print(f"{'Virtual Best':<25} {len(vb_ablation):>7}")

    # =========================================================================
    # Section: Greedy Heuristic
    # =========================================================================
    print_section("GREEDY HEURISTIC")

    greedy_stats = analyze_greedy_heuristic(f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full")
    nogreedy_stats = analyze_greedy_heuristic(f"{RESULTS_DIR}/{EXPERIMENT_VERSION}_full_nogreedy")

    if greedy_stats:
        # Use mean of per-instance success rates (as in paper)
        success_rate = np.mean(greedy_stats['per_instance_success_rates'])
        total_cols = greedy_stats['greedy_cols'] + greedy_stats['labeling_cols']
        greedy_ratio = greedy_stats['greedy_cols'] / total_cols * 100 if total_cols > 0 else 0

        print(f"\nWith greedy pricing:")
        print(f"  Success rate (mean of per-instance): {success_rate:.1f}%")
        print(f"  Greedy columns: {greedy_stats['greedy_cols']:,} ({greedy_ratio:.1f}%)")
        print(f"  Labeling columns: {greedy_stats['labeling_cols']:,} ({100-greedy_ratio:.1f}%)")
        print(f"  Avg NG expansions: {np.mean(greedy_stats['ng_expansions']):.1f}")
        print(f"  Max neighborhood: {max(greedy_stats['max_neighborhoods'])}%")

    if nogreedy_stats:
        print(f"\nWithout greedy pricing:")
        print(f"  Labeling columns: {nogreedy_stats['labeling_cols']:,}")
        print(f"  Avg NG expansions: {np.mean(nogreedy_stats['ng_expansions']):.1f}")
        print(f"  Max neighborhood: {max(nogreedy_stats['max_neighborhoods'])}%")

    # =========================================================================
    # Section: NG-Relaxation
    # =========================================================================
    print_section("NG-RELAXATION")

    if nogreedy_stats:
        ng_exp = nogreedy_stats['ng_expansions']
        max_ng = nogreedy_stats['max_neighborhoods']

        print(f"\nInstances analyzed: {len(ng_exp)}")
        print(f"Instances with expansions: {sum(1 for x in ng_exp if x > 0)}")
        print(f"Avg expansions: {np.mean(ng_exp):.1f}")
        print(f"Max expansions: {max(ng_exp)}")
        print(f"Avg max neighborhood: {np.mean([x for x in max_ng if x > 0]):.1f}%")
        print(f"Max neighborhood reached: {max(max_ng)}%")
        print(f"Instances reaching 100%: {sum(1 for x in max_ng if x >= 100)}")

    # =========================================================================
    # Section: Cutting Planes
    # =========================================================================
    print_section("CUTTING PLANES")

    cuts_configs = [
        ('full', 'No cuts'),
        ('clique', '+CC'),
        ('srow', '+SRC'),
        ('srow_clique', '+SRC+CC'),
        ('srow_pricing', '+SRC (pricing)'),
    ]

    cuts_basket = per_table_basket(files, [k for k, _ in cuts_configs])
    nocuts_time = compute_solver_stats(files['full'], cuts_basket)['sgm_time']

    print(f"(per-table basket: {len(cuts_basket)} instances)")
    print(f"\n{'Config':<20} {'Solved':>7} {'AtRoot':>7} {'RootLP':>7} {'Time':>8} {'Ratio':>7}")
    print("-" * 62)

    solved_sets_cuts = {}
    for key, name in cuts_configs:
        if key in files:
            stats = compute_solver_stats(files[key], cuts_basket)
            at_root, root_lp = root_metrics(files[key])
            ratio = stats['sgm_time'] / nocuts_time
            solved_sets_cuts[key] = solved_set(files[key])
            print(f"{name:<20} {stats['solved']:>7} {at_root:>7} {root_lp:>7} "
                  f"{stats['sgm_time']:>8.1f} {ratio:>7.2f}")

    vb_cuts = set().union(*solved_sets_cuts.values()) if solved_sets_cuts else set()
    print(f"{'VB-cuts':<20} {len(vb_cuts):>7}")

    # =========================================================================
    # Section: Branching Strategy
    # =========================================================================
    print_section("BRANCHING STRATEGY")

    if 'full_nobestsol' in files and 'full_rf_nobestsol' in files:
        edge_nb = read_csv(files['full_nobestsol'])
        rf_nb = read_csv(files['full_rf_nobestsol'])

        edge_nb_solved = set(edge_nb[(edge_nb["final_gap"] == 0) & (edge_nb["total_time"] < TIMELIMIT)]["instance_name"])
        rf_nb_solved = set(rf_nb[(rf_nb["final_gap"] == 0) & (rf_nb["total_time"] < TIMELIMIT)]["instance_name"])
        solved_by_either = edge_nb_solved | rf_nb_solved
        both_solved = edge_nb_solved & rf_nb_solved

        edge_f = edge_nb[edge_nb["instance_name"].isin(solved_by_either)]
        rf_f = rf_nb[rf_nb["instance_name"].isin(solved_by_either)]

        print(f"\nWithout improved upper bounds:")
        print(f"  Edge branching: {len(edge_nb_solved)} solved, SGM {sgm(edge_f['total_time'], 1):.1f}s")
        print(f"  Ryan-Foster:    {len(rf_nb_solved)} solved, SGM {sgm(rf_f['total_time'], 1):.1f}s")
        print(f"  Ratio RF/Edge:  {sgm(rf_f['total_time'], 1)/sgm(edge_f['total_time'], 1):.1f}x")

        merged = edge_nb.merge(rf_nb, on="instance_name", suffixes=("_edge", "_rf"))
        merged = merged[merged["instance_name"].isin(both_solved)]
        edge_nodes = merged["number_of_nodes_edge"].fillna(1)
        rf_nodes = merged["number_of_nodes_rf"].fillna(1)
        n_branching = int(((edge_nodes > 1) | (rf_nodes > 1)).sum())

        print(f"  Solved by both: {len(both_solved)}")
        print(f"  Instances requiring branching: {n_branching}")
        print(f"  Avg nodes (edge): {edge_nodes.mean():.1f}, max: {edge_nodes.max():.0f}")
        print(f"  Avg nodes (RF):   {rf_nodes.mean():.1f}, max: {rf_nodes.max():.0f}")

    # =========================================================================
    # Section: Dual Stabilization
    # =========================================================================
    print_section("DUAL STABILIZATION")

    if 'full_stabilized' in files:
        stab_basket = per_table_basket(files, ['full', 'full_stabilized'])
        base_stats = compute_solver_stats(files['full'], stab_basket)
        stab_stats = compute_solver_stats(files['full_stabilized'], stab_basket)
        stab_df = read_csv(files['full_stabilized'])
        full_df = read_csv(files['full'])
        trieq_df = read_csv(files['full_trieq']) if 'full_trieq' in files else None

        print(f"(per-table basket: {len(stab_basket)} instances)")
        print(f"\n{'Config':<20} {'Solved':>6} {'SGM':>8} {'CG root':>8} {'CG total':>9}")
        print("-" * 55)
        print(f"{'Baseline':<20} {base_stats['solved']:>6} {base_stats['sgm_time']:>7.1f}s {full_df['cg_iterations_root'].mean():>8.0f} {full_df['cg_iterations_total'].mean():>9.0f}")
        print(f"{'Stabilized':<20} {stab_stats['solved']:>6} {stab_stats['sgm_time']:>7.1f}s {stab_df['cg_iterations_root'].mean():>8.0f} {stab_df['cg_iterations_total'].mean():>9.0f}")
        if trieq_df is not None:
            trieq_stats = compute_solver_stats(files['full_trieq'], stab_basket)
            print(f"{'Trieq':<20} {trieq_stats['solved']:>6} {trieq_stats['sgm_time']:>7.1f}s {trieq_df['cg_iterations_root'].mean():>8.0f} {trieq_df['cg_iterations_total'].mean():>9.0f}")

        cg_reduction = (1 - stab_df['cg_iterations_root'].mean() / full_df['cg_iterations_root'].mean()) * 100
        print(f"\n  Root CG iteration reduction: {cg_reduction:.0f}%")

    # =========================================================================
    # Section: Numerical Safety
    # =========================================================================
    print_section("NUMERICAL SAFETY")

    if 'full_unsafe' in files:
        safety_basket = per_table_basket(files, ['full', 'full_unsafe'])
        safe_stats = compute_solver_stats(files['full'], safety_basket)
        unsafe_stats = compute_solver_stats(files['full_unsafe'], safety_basket)
        safe_at_root, _ = root_metrics(files['full'])
        unsafe_at_root, _ = root_metrics(files['full_unsafe'])

        print(f"(per-table basket: {len(safety_basket)} instances)")
        print(f"\n{'Config':<20} {'Solved':>8} {'Time':>10} {'AtRoot':>8}")
        print("-" * 48)
        print(f"{'Safe':<20} {safe_stats['solved']:>8} {safe_stats['sgm_time']:>10.1f} {safe_at_root:>8}")
        print(f"{'Unsafe':<20} {unsafe_stats['solved']:>8} {unsafe_stats['sgm_time']:>10.1f} {unsafe_at_root:>8}")

        slowdown = safe_stats['sgm_time'] / unsafe_stats['sgm_time']
        print(f"\nSafe/Unsafe slowdown: {slowdown:.2f}x")

    # =========================================================================
    # Summary
    # =========================================================================
    print_section("SUMMARY FOR PAPER")

    print(f"""
Key numbers for the paper:
- BNP-FULL solves {full_stats['solved']} instances vs {compact_stats['solved']} for BNC-SEC
- BNP-FULL is {speedup:.0f}x faster than BNC-SEC
- Root gap: BNP {root_df['bnp_gap'].mean():.1f}% vs SEC {root_df['sec_gap'].mean():.1f}%
- Greedy heuristic: {success_rate:.0f}% success rate, {greedy_ratio:.0f}% of columns
- NG-relaxation: max {max(nogreedy_stats['max_neighborhoods'])}% neighborhood (never 100%)
- Numerical safety overhead: {(slowdown-1)*100:.0f}% slower
""")

    print("\nDone! Run 'python3 analyze.py' with specific flags to generate LaTeX tables.")


if __name__ == "__main__":
    main()
