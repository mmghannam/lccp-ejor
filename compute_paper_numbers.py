#!/usr/bin/env python3
"""
Reproduce every number reported in the computational study (Section 7) of the
LCCP branch-price-and-cut paper directly from the per-instance result CSVs in
``results/``.

This script is intentionally self-contained: it depends only on ``pandas`` and
``numpy`` (see requirements.txt) and does not need the raw solver logs. Run it
from this directory:

    python3 compute_paper_numbers.py

Each printed block is annotated with the table / sentence in the paper it
reproduces. The two greedy-pricing / ng-relaxation prose numbers (success rate,
neighbourhood expansions) are parsed from the raw ``.sol`` logs and are produced
by ``generate_paper_data.py`` instead (the logs are not shipped here).

Configuration -> CSV mapping
----------------------------
    bnc-sec          compact.csv            (Gurobi SEC, MCV warm start)
    bnc-full         compact_bks.csv        (Gurobi SEC, best-known-solution warm start)
    bnp-mcv          27_full_nobestsol.csv  (branch-and-price, MCV warm start)
    bnp-full         27_full.csv            (branch-and-price, best-known-solution warm start)
    ablation/cuts    27_full_<feature>.csv  (one feature toggled)
"""

import math
import os

import numpy as np
import pandas as pd

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TIMELIMIT = 2 * 3600  # 2 hours


# --------------------------------------------------------------------------- #
# Helpers (mirror analyze.py: read_csv / fix_time / sgm)
# --------------------------------------------------------------------------- #
def fix_time(t):
    """Unparseable / NaN solving time -> a value just past the time limit."""
    try:
        x = float(t)
        return 7230.0 if math.isnan(x) else x
    except (TypeError, ValueError):
        return 7230.0


def read_csv(name):
    path = os.path.join(RESULTS, name)
    df = pd.read_csv(path)
    df["total_time"] = df["total_time"].map(fix_time)
    if "compact" not in name:
        # True optimality gap (final dual bound) -> determines solved status.
        df["final_gap"] = (df["primal_bound"] - df["dual_bound"]) / df["primal_bound"]
    # Root gap (root dual bound). For compact, final_gap is the parsed Gurobi gap.
    df["root_gap"] = (df["primal_bound"] - df["dual_bound_at_root"]) / df["primal_bound"]
    df["instance_name"] = df["instance_name"].astype(str)
    return df


def sgm(values, shift=1):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return float("nan")
    return float(np.exp(np.sum(np.log(v + shift)) / len(v)) - shift)


def solved_set(df):
    return set(df[(df["final_gap"] == 0) & (df["total_time"] < TIMELIMIT)]["instance_name"])


def sub(df, basket):
    return df[df["instance_name"].isin(basket)]


def at_root(df):
    return int(((df["final_gap"] == 0) & (df["total_time"] < TIMELIMIT) & (df["number_of_nodes"] == 1)).sum())


def root_lp_converged(df):
    solved = (df["final_gap"] == 0) & (df["total_time"] < TIMELIMIT)
    return set(df[(df["dual_bound_at_root"] > 1) | solved]["instance_name"])


def hr(title):
    print("\n" + "=" * 72 + f"\n {title}\n" + "=" * 72)


# --------------------------------------------------------------------------- #
def main():
    full = read_csv("27_full.csv")            # bnp-full
    mcv = read_csv("27_full_nobestsol.csv")   # bnp-mcv
    sec = read_csv("compact.csv")             # bnc-sec
    secb = read_csv("compact_bks.csv")        # bnc-full

    S = {k: solved_set(v) for k, v in
         {"bnc-sec": sec, "bnp-mcv": mcv, "bnp-full": full, "bnc-full": secb}.items()}

    # z* = best known integer optimum per instance (min primal across configs).
    zstar = pd.concat([full.set_index("instance_name")["primal_bound"],
                       mcv.set_index("instance_name")["primal_bound"],
                       sec.set_index("instance_name")["primal_bound"],
                       secb.set_index("instance_name")["primal_bound"]], axis=1).min(axis=1)
    converged = root_lp_converged(full)  # 52 instances where bnp-full's root LP converges

    def zstar_root_gap(df):
        d = df.set_index("instance_name")
        idx = [i for i in converged if i in d.index]
        z = zstar.loc[idx]
        gap = ((z - d.loc[idx, "dual_bound_at_root"]) / z).clip(lower=0)
        return gap.mean() * 100

    # ---- Table 1: Overall performance (dual side) ------------------------- #
    hr("TABLE 1  Overall performance (dual side)")
    basket = S["bnc-sec"] | S["bnp-mcv"] | S["bnp-full"] | S["bnc-full"]
    print(f"per-table basket (solved by any exact config): {len(basket)} instances")
    print(f"{'config':<10}{'Solved':>8}{'Gap %':>8}{'Time':>10}")
    for k, df in [("bnc-sec", sec), ("bnp-mcv", mcv), ("bnc-full", secb), ("bnp-full", full)]:
        gap = zstar_root_gap(df)
        print(f"{k:<10}{len(S[k]):>8}{gap:>8.1f}{sgm(sub(df, basket)['total_time']):>10.1f}")
    t = lambda df: sgm(sub(df, basket)["total_time"])
    print(f"\nspeedup bnc-sec / bnp-full  : {t(sec)/t(full):.1f}x")
    print(f"speedup bnc-sec / bnp-mcv   : {t(sec)/t(mcv):.1f}x   (both MCV warm start)")
    print(f"speedup bnc-full / bnp-full : {t(secb)/t(full):.1f}x  (both best-known warm start)")

    # ---- Headline figures ------------------------------------------------- #
    hr("HEADLINE")
    def nnodes(name):
        m = __import__("re").search(r"(\d+)$", name)
        return int(m.group(1)) if m else 0
    print(f"bnp-full solves {len(S['bnp-full'])} instances (largest {max(map(nnodes, S['bnp-full']))} nodes)")
    print(f"bnp-full closes {len(S['bnp-full'] - S['bnc-full'])} instances unsolved by the previous baseline (bnc-full)")
    print(f"more than bnc-sec: {len(S['bnp-full']) - len(S['bnc-sec'])}")

    # ---- Table 2: Ablation ------------------------------------------------ #
    hr("TABLE 2  Ablation study")
    abl = [("bnp-full", "27_full.csv"), ("nosymbr", "27_full_nosymbr.csv"),
           ("nobidir", "27_full_nobidir.csv"), ("nopar", "27_full_nopar.csv"),
           ("noearly", "27_full_noearly.csv"), ("nogreedy", "27_full_nogreedy.csv"),
           ("nobestsol", "27_full_nobestsol.csv"), ("withtrieq", "27_full_trieq.csv")]
    dfs = {name: read_csv(csv) for name, csv in abl}
    abasket = set().union(*[solved_set(v) for v in dfs.values()])
    ft = sgm(sub(dfs["bnp-full"], abasket)["total_time"])
    print(f"per-table basket: {len(abasket)} instances")
    print(f"{'config':<12}{'Solved':>7}{'AtRoot':>7}{'RootLP':>7}{'Time':>8}{'Ratio':>7}{'Gap %':>7}")
    vb = set()
    vb_mins = {}
    for name, _ in abl:
        df = dfs[name]
        tt = sgm(sub(df, abasket)["total_time"])
        gap = sub(df, abasket)["root_gap"].clip(lower=0).mean() * 100
        print(f"{name:<12}{len(solved_set(df)):>7}{at_root(df):>7}{len(root_lp_converged(df)):>7}"
              f"{tt:>8.1f}{tt/ft:>7.2f}{gap:>7.1f}")
        vb |= solved_set(df)
        for inst in abasket:
            r = df[df["instance_name"] == inst]
            if len(r):
                vb_mins[inst] = min(vb_mins.get(inst, 1e18), fix_time(r.iloc[0]["total_time"]))
    vbt = sgm(list(vb_mins.values()))
    print(f"{'Virtual Best':<12}{len(vb):>7}{'':>7}{'':>7}{vbt:>8.1f}{vbt/ft:>7.2f}")

    # ---- Table: Cutting planes ------------------------------------------- #
    hr("TABLE  Branch-price-and-cut (cutting planes)")
    cuts = [("bnp-full", "27_full.csv"), ("withcc", "27_clique.csv"),
            ("withsrc", "27_srow.csv"), ("withsrc+cc", "27_srow_clique.csv"),
            ("withsrc+pricing", "27_srow_pricing.csv")]
    cdfs = {name: read_csv(csv) for name, csv in cuts}
    cbasket = set().union(*[solved_set(v) for v in cdfs.values()])
    cft = sgm(sub(cdfs["bnp-full"], cbasket)["total_time"])
    print(f"per-table basket: {len(cbasket)} instances")
    print(f"{'config':<16}{'Solved':>7}{'AtRoot':>7}{'RootLP':>7}{'Time':>8}{'Ratio':>7}")
    cvb, cvb_mins = set(), {}
    for name, _ in cuts:
        df = cdfs[name]
        tt = sgm(sub(df, cbasket)["total_time"])
        print(f"{name:<16}{len(solved_set(df)):>7}{at_root(df):>7}{len(root_lp_converged(df)):>7}"
              f"{tt:>8.1f}{tt/cft:>7.2f}")
        cvb |= solved_set(df)
        for inst in cbasket:
            r = df[df["instance_name"] == inst]
            if len(r):
                cvb_mins[inst] = min(cvb_mins.get(inst, 1e18), fix_time(r.iloc[0]["total_time"]))
    cvbt = sgm(list(cvb_mins.values()))
    print(f"{'Virtual Best':<16}{len(cvb):>7}{'':>7}{'':>7}{cvbt:>8.1f}{cvbt/cft:>7.2f}")

    # ---- Table: Numerical safety ----------------------------------------- #
    hr("TABLE  Cost of numerical safety")
    unsafe = read_csv("27_full_unsafe.csv")
    nbasket = solved_set(full) | solved_set(unsafe)
    for name, df in [("bnp-full (safe)", full), ("bnp-unsafe", unsafe)]:
        print(f"{name:<18} Solved {len(solved_set(df)):>3}  Time {sgm(sub(df, nbasket)['total_time']):>6.1f}  AtRoot {at_root(df):>3}")
    print(f"safe / unsafe slowdown: {sgm(sub(full, nbasket)['total_time'])/sgm(sub(unsafe, nbasket)['total_time']):.2f}x")
    print(f"unsafe closes (not safe): {sorted(solved_set(unsafe) - solved_set(full))}")
    print(f"safe closes (not unsafe): {sorted(solved_set(full) - solved_set(unsafe))}")

    # ---- Dual stabilization ---------------------------------------------- #
    hr("Dual stabilization")
    stab = read_csv("27_full_stabilized.csv")
    print(f"CG root : full={full['cg_iterations_root'].mean():.0f}  stab={stab['cg_iterations_root'].mean():.0f}"
          f"  ({(1-stab['cg_iterations_root'].mean()/full['cg_iterations_root'].mean())*100:.0f}% fewer)")
    print(f"CG total: full={full['cg_iterations_total'].mean():.0f}  stab={stab['cg_iterations_total'].mean():.0f}"
          f"  ({(1-stab['cg_iterations_total'].mean()/full['cg_iterations_total'].mean())*100:.0f}% fewer)")
    sbasket = solved_set(full) | solved_set(stab)
    print(f"SGM: full={sgm(sub(full, sbasket)['total_time']):.1f}  stab={sgm(sub(stab, sbasket)['total_time']):.1f}"
          f"   solved: full={len(solved_set(full))}  stab={len(solved_set(stab))}")

    # ---- Branching strategy ---------------------------------------------- #
    hr("Branching strategy (no improved upper bounds)")
    edge = read_csv("27_full_nobestsol.csv")
    rf = read_csv("27_full_rf_nobestsol.csv")
    es, rs = solved_set(edge), solved_set(rf)
    both = es & rs
    either = es | rs
    print(f"edge: {len(es)} solved, SGM {sgm(sub(edge, either)['total_time']):.1f}s")
    print(f"RF  : {len(rs)} solved, SGM {sgm(sub(rf, either)['total_time']):.1f}s"
          f"   (RF/edge = {sgm(sub(rf, either)['total_time'])/sgm(sub(edge, either)['total_time']):.1f}x)")
    m = edge.merge(rf, on="instance_name", suffixes=("_e", "_r"))
    m = m[m["instance_name"].isin(both)]
    en, rn = m["number_of_nodes_e"].fillna(1), m["number_of_nodes_r"].fillna(1)
    print(f"solved by both: {len(both)}  requiring branching: {int(((en>1)|(rn>1)).sum())}")
    print(f"edge nodes avg {en.mean():.1f} (max {en.max():.0f}); RF nodes avg {rn.mean():.1f} (max {rn.max():.0f})")


if __name__ == "__main__":
    main()
