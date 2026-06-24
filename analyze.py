from cmath import isnan
from collections import defaultdict
import argparse
import glob
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_latest_experiment_files():
    """
    Automatically finds the latest experiment files in the results directory.
    Returns a list of CSV file paths for the highest numbered experiment.
    Always includes compact.csv as a baseline.
    """
    results_dir = "results"
    csv_files = glob.glob(os.path.join(results_dir, "*.csv"))

    # Extract experiment numbers from filenames (e.g., "24_full.csv" -> 24)
    experiment_numbers = set()
    for f in csv_files:
        basename = os.path.basename(f)
        match = re.match(r'^(\d+)_', basename)
        if match:
            experiment_numbers.add(int(match.group(1)))

    if not experiment_numbers:
        return ["results/compact.csv"]

    # Get the highest experiment number
    latest_num = max(experiment_numbers)

    # Find all files for the latest experiment
    latest_files = sorted([f for f in csv_files if os.path.basename(f).startswith(f"{latest_num}_")])

    # Always include compact.csv as baseline if it exists
    compact_path = os.path.join(results_dir, "compact.csv")
    result = []
    if os.path.exists(compact_path):
        result.append(compact_path)
    result.extend(latest_files)

    return result

def extract_number_from_filename(filename):
    import re
    match = re.search(r'(\d+)(?!.*\d)', filename)
    if match:
        return int(match.group(1))
    return None

def sgm(numbers, shift):
    if len(numbers) == 0:
        return float('nan')
    return np.exp(np.sum(np.log((numbers + shift))) / len(numbers)) - shift


def latex_table(files, optimal_counts, avg_times, avg_nodes):
    print("\\begin{table}")
    print("\\centering")
    print("\\begin{tabular}{|c|c|c|c|}")
    print("\\hline")
    print("File & Optimal Instances & Average Solving Time (s) & Average Nodes Explored \\\\")
    print("\\hline")
    for i, file in enumerate(files):
        print(
            f"{file} & {optimal_counts[i]} / {len(all_instances)} & {avg_times[i]:.2f} & {avg_nodes[i]:.2f} \\\\")
        print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Results summary}")
    print("\\end{table}")


def bar_time(files, avg_times):
    import plotly.express as px

    files = [file_name_map[x] for x in files]
    fig = px.bar(x=files, y=avg_times, color=files, log_y=True)

    # fig.update_yaxes(title_text="average time", range=[0,1500])
    # fig.update_xaxes(title_text="solver", range=[-0.5,4.5])
    # fig.update_layout(
    #     font=dict(
    #         size=24
    #     )
    # )
    fig.show()



def instance_table(): 
    compact_df = pd.read_csv("results/compact.csv")
    bnp_df = pd.read_csv("results/bnp1701.csv")
    bnp_df.sort_values(by=["instance_name"], inplace=True)
    compact_df.sort_values(by=["instance_name"], inplace=True)
    

    # bnp_df["dual_bound_at_root"] = bnp_df["dual_bound_at_root"].map(lambda x: f"{x:.0f}")
    # bnp_df["primal_bound"] = bnp_df["primal_bound"].map(lambda x: f"{x:.0f}")
    # bnp_df["dual_bound"] = bnp_df["dual_bound"].map(lambda x: f"{x:.0f}")
    # bnp_df["final_gap"] = bnp_df["final_gap"].map(lambda x: f"{x:.0f}")

    bnp_df["total_time"] = bnp_df["total_time"].map(lambda x: "TL" if x >= 10600 or str(x).lower() == "nan" else f"{max(0.01, x):.2f}")
    bnp_df["final_gap"] = (bnp_df["primal_bound"] - bnp_df["dual_bound_at_root"] ) * 100/ bnp_df["primal_bound"]
    # print(list(bnp_df["final_gap"]))
    compact_df["total_time"] = compact_df["total_time"].map(lambda x: "TL" if x >= 10600 or str(x).lower() == "nan" else f"{x:.2f}")

    new_df = bnp_df.merge(compact_df, on=["instance_name"] , suffixes=("_bnp", "_compact"))
    new_df["input_nodes"] = new_df["instance_name"].map(lambda x: extract_number_from_filename(x)) 

    new_df = new_df[[
        "instance_name",
        "input_nodes",
        "dual_bound_compact",
        "primal_bound_compact", 
        "final_gap_compact",
        "total_time_compact",
        "dual_bound_at_root_bnp", "dual_bound_bnp", 
        "primal_bound_bnp", 
        "final_gap_bnp",
        "total_time_bnp",
        ]]

    new_df["instance_name"] = new_df["instance_name"].map(lambda x: f"\\texttt{{{x}}}".replace("_", "\\_"))

    # new_df = new_df [ new_df["final_gap_compact"] == 0]
    # add compact columns
    
    # print(bnp_df)
    new_df["total_time_bnp"] = new_df["total_time_bnp"].map(lambda x: 10600 if x == "TL" else float(x))
    # new_df.sort_values(by="total_time_bnp", ascending=True, inplace=True)
    new_df.sort_values(by="input_nodes", ascending=True, inplace=True)
    new_df["total_time_bnp"] = new_df["total_time_bnp"].map(lambda x: "TL" if x == 10600 else f"{x:.2f}")
    new_df.columns = ["Instance", "Nodes",
                        "DB", "PB", "Gap", "Time(s)",
                        "DBr", "DB", "PB", "Gap", "Time(s)",
                       ]
    new_df.columns = pd.MultiIndex.from_tuples([("Instance", ""),
                                                ("Nodes", ""),
                                                ("\\texttt{compact}", "DB"),
                                                ("\\texttt{compact}", "PB"),
                                                ("\\texttt{compact}", "Gap"),
                                                ("\\texttt{compact}", "Time(s)"),
                                                ("\\texttt{bnp}", "DBroot"),
                                                ("\\texttt{bnp}", "DB"),
                                                ("\\texttt{bnp}", "PB"),
                                                ("\\texttt{bnp}", "Gap"),
                                                ("\\texttt{bnp}", "Time(s)"),
                                                ])
    new_df.to_latex(
        "~/papers/lccp-paper/all-instances.tex",
        index=False,
        longtable=True,
        float_format="{:.1f}".format,
        column_format="|l|c|" + "r" * 4+ "|" + "r" * 5 + "|",
        multicolumn_format="c|",
    )


def aggregated_table(name, *columns):

    data = zip(*columns)
    df = pd.DataFrame(data)
    df.columns = ["Solver", "Time", "Ratio", "Solved"]
    
    df.to_latex(
        f"~/papers/lccp-paper/aggregated-{name}.tex",
        index=False,
        # longtable=True,
        float_format="{:.2f}".format,
        # column_format="|l" + "r" * (len(columns) - 1)+ "|",
        # multicolumn_format="c|",
    )

def plot_solved(path=None):
        flat_solved_data = []
        for i, solver in enumerate(solvers):
            for tl in solved_count:
                flat_solved_data.append([solver, tl, solved_count[tl][i]])

        df = pd.DataFrame(flat_solved_data, columns=["solver", "time", "instances"])

        import plotly.express as px
        
        import plotly.graph_objects as go

        # Create a line plot
        fig = go.Figure()
        for solver in solvers:    
            # Add a line trace with customizations
            solver_df = df[df.solver == solver]
            if solver in ["compact", "bnp", "bnp-basic"]:
                style = dict(width=2.5)
            else:
                style = dict(width=2, dash="dashdot")
            

            if solver == "compact": solver = "bnc-sec"
            if solver == "bnp": 
                solver = "bnp-full"
                style["color"] = "green"
            if solver == "bnp-nopar":
                style["color"] = "brown"
            if solver == "bnp-noint":
                solver = "bnp-noearly"
            
            solver = solver.upper()
            fig.add_trace(go.Scatter(x=[math.log(x) for x in solver_df.time], y=solver_df.instances, mode='lines',
                                    line=style,
                                    name=solver,
                                    ))
            
        tickvals = [ 1, 10, 1000, 3600, 10600]
        fig.update_layout(
        # title='Plot with Logarithmic Y-Axis',
        # xaxis_title='X Axis',
        xaxis=dict(
            # title='Time [s] (log scale)',
            linecolor="black",
            # type='log' ,
            #  range=[0, math.log(10600)],
            tickvals=[math.log(x) for x in tickvals],
            ticktext=list(map(str, tickvals))
        ),
        yaxis=dict(
            # title='Instances solved',
            linecolor="black",
            tickvals=list(range(0, 51, 10)),
            range=[0, 55],
        ),
        # bigger font 
        font=dict(
            size=18,
            # family="Common Serif",
            family="Computer Modern",
            color="black"
        ),
        # paper_bgcolor='white',  # Sets the background color of the paper (outside the plot)
        plot_bgcolor='white' 
        )


        # fig.add_annotation(dict(
        #     x=math.log(10600),
        #     y=0,
        #     ax=math.log(10600),
        #     ay=0,
        #     xref='x',
        #     yref='y',
        #     showarrow=True,
        #     arrowhead=3,
        #     arrowsize=2,
        #     # arrowwidth=2,
        #     arrowcolor='black'
        # ))
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray',
                         griddash='dashdot')

        if path:
            fig.write_image(path)
        else:
            fig.show()

def gap_at_root_compare():
    compact = pd.read_csv("results/compact.csv")
    bnp = pd.read_csv("results/bnp0304.csv")

    df = pd.concat([compact, bnp])
    bks = best_known_sols(df)

    bnp["final_gap"] = (bnp["primal_bound"] - bnp["dual_bound_at_root"]) / bnp["primal_bound"]
    instances_roots_solved_of_bnp = bnp[(bnp["dual_bound_at_root"] > 1) | ((bnp["final_gap"] == 0) & (bnp["total_time"] < timelimit + 20)) ]["instance_name"].unique()
    print("bnp solved", len(instances_roots_solved_of_bnp))
    bnp = bnp[bnp["instance_name"].isin(instances_roots_solved_of_bnp)]
    compact = compact[compact["instance_name"].isin(instances_roots_solved_of_bnp)]

    # compact["gap_at_root"] = (compact["primal_bound_at_root"] - compact["dual_bound_at_root"]) / compact["primal_bound_at_root"]
    # bnp["gap_at_root"] = (bnp["primal_bound_at_root"] - bnp["dual_bound_at_root"]) / bnp["primal_bound_at_root"]
    compact_gaps = {}
    bnp_gaps = {}
    for instance, solval in bks.items():
        print(instance)
        if len(compact[compact["instance_name"] == instance]) == 0:
            continue
        compact_dual = compact[compact["instance_name"] == instance]["dual_bound_at_root"].iloc[0]
        print(compact_dual)
        bnp_dual = bnp[bnp["instance_name"] == instance]["dual_bound_at_root"].iloc[0]
        compact_gaps[instance] = (solval - compact_dual) / solval
        bnp_gaps[instance] = (solval - bnp_dual) / solval

    assert(len(compact_gaps) == len(bnp_gaps))
    print(sum(compact_gaps.values()) / len(compact_gaps))
    print(sum(bnp_gaps.values()) / len(bnp_gaps))

def root_bound_comparison(compact_file="results/compact.csv", bnp_file="results/27_full.csv"):
    """
    Compare root dual bounds between SEC (compact) and BNP (set partitioning) formulations.
    Prints statistics for the paper's LP Relaxations section.
    """
    compact = pd.read_csv(compact_file)
    bnp = pd.read_csv(bnp_file)

    # Get best known solutions
    all_df = pd.concat([compact, bnp])
    bks = best_known_sols(all_df)

    # Filter to instances where BNP converged at root (has valid dual bound)
    bnp_valid = bnp[(bnp['dual_bound_at_root'] > 0)]

    # Compute gaps
    results = []
    for instance in bnp_valid.instance_name.unique():
        if instance not in compact.instance_name.values:
            continue

        opt = bks[instance]
        bnp_db = bnp[bnp.instance_name == instance]['dual_bound_at_root'].values[0]
        sec_db = compact[compact.instance_name == instance]['dual_bound_at_root'].values[0]

        bnp_gap = (opt - bnp_db) / opt * 100 if opt > 0 else 0
        sec_gap = (opt - sec_db) / opt * 100 if opt > 0 else 0

        # Check if ceil(bnp_db) >= opt - 1
        bnp_tight = np.ceil(bnp_db) >= opt - 1

        results.append({
            'instance': instance,
            'opt': opt,
            'bnp_db': bnp_db,
            'sec_db': sec_db,
            'bnp_gap': bnp_gap,
            'sec_gap': sec_gap,
            'bnp_tight': bnp_tight,
            'bnp_better': bnp_db > sec_db
        })

    df = pd.DataFrame(results)

    print(f'Root Bound Comparison')
    print(f'=' * 50)
    print(f'Instances analyzed: {len(df)}')
    print(f'')
    print(f'Average root gap (BNP): {df.bnp_gap.mean():.2f}%')
    print(f'Average root gap (SEC): {df.sec_gap.mean():.2f}%')
    print(f'')
    print(f'Instances where ceil(BNP root) >= opt-1: {df.bnp_tight.sum()} / {len(df)}')
    print(f'Instances where BNP root > SEC root: {df.bnp_better.sum()} / {len(df)}')
    print(f'')
    print(f'Max BNP gap: {df.bnp_gap.max():.2f}%')
    print(f'Max SEC gap: {df.sec_gap.max():.2f}%')

    return df

def generate_supplementary_tables(compact_file="results/compact.csv", bnp_file="results/27_full.csv",
                                   output_dir="~/papers/lccp-paper/EJOR Version"):
    """
    Generate comprehensive per-instance results tables for EJOR supplementary material.
    Creates both LaTeX and CSV versions.
    """
    import os

    output_dir = os.path.expanduser(output_dir)
    timelimit = 2 * 3600

    # Read data
    compact = pd.read_csv(compact_file)
    bnp = pd.read_csv(bnp_file)

    # Fix times
    compact["total_time"] = compact["total_time"].map(fix_time)
    bnp["total_time"] = bnp["total_time"].map(fix_time)

    # Compute gaps for BNP
    bnp["final_gap"] = (bnp["primal_bound"] - bnp["dual_bound"]) / bnp["primal_bound"] * 100
    bnp["final_gap"] = bnp["final_gap"].fillna(100)

    # Get best known solutions
    all_df = pd.concat([compact, bnp])
    bks = best_known_sols(all_df)

    # Get all instances
    all_instances = sorted(set(compact.instance_name.unique()) | set(bnp.instance_name.unique()),
                          key=lambda x: (extract_number_from_filename(x) or 0, x))

    results = []
    for instance in all_instances:
        nodes = extract_number_from_filename(instance)
        best = bks.get(instance, float('nan'))

        # BNP results
        bnp_row = bnp[bnp.instance_name == instance]
        if len(bnp_row) > 0:
            bnp_row = bnp_row.iloc[0]
            bnp_pb = bnp_row['primal_bound'] if not pd.isna(bnp_row['primal_bound']) else '-'
            bnp_db_root = bnp_row['dual_bound_at_root'] if not pd.isna(bnp_row['dual_bound_at_root']) else '-'
            bnp_db = bnp_row['dual_bound'] if not pd.isna(bnp_row['dual_bound']) else '-'
            bnp_gap = bnp_row['final_gap'] if not pd.isna(bnp_row['final_gap']) else 100
            bnp_time = bnp_row['total_time']
            bnp_nodes = bnp_row['number_of_nodes'] if not pd.isna(bnp_row['number_of_nodes']) else '-'
            bnp_status = 'opt' if bnp_gap == 0 and bnp_time < timelimit else ('TL' if bnp_time >= timelimit else 'open')
        else:
            bnp_pb = bnp_db_root = bnp_db = bnp_gap = bnp_time = bnp_nodes = '-'
            bnp_status = '-'

        # SEC (compact) results
        sec_row = compact[compact.instance_name == instance]
        if len(sec_row) > 0:
            sec_row = sec_row.iloc[0]
            sec_pb = sec_row['primal_bound'] if not pd.isna(sec_row['primal_bound']) else '-'
            sec_db = sec_row['dual_bound'] if not pd.isna(sec_row['dual_bound']) else '-'
            sec_gap = sec_row['final_gap'] if not pd.isna(sec_row['final_gap']) else 100
            sec_time = sec_row['total_time']
            sec_nodes_val = sec_row['number_of_nodes'] if not pd.isna(sec_row['number_of_nodes']) else '-'
            sec_status = 'opt' if sec_gap == 0 and sec_time < timelimit else ('TL' if sec_time >= timelimit else 'open')
        else:
            sec_pb = sec_db = sec_gap = sec_time = sec_nodes_val = '-'
            sec_status = '-'

        results.append({
            'Instance': instance,
            'n': nodes,
            'BKS': best,
            'BNP_PB': bnp_pb,
            'BNP_DB_root': bnp_db_root,
            'BNP_DB': bnp_db,
            'BNP_Gap': bnp_gap,
            'BNP_Time': bnp_time,
            'BNP_Nodes': bnp_nodes,
            'BNP_Status': bnp_status,
            'SEC_PB': sec_pb,
            'SEC_DB': sec_db,
            'SEC_Gap': sec_gap,
            'SEC_Time': sec_time,
            'SEC_Nodes': sec_nodes_val,
            'SEC_Status': sec_status,
        })

    df = pd.DataFrame(results)

    # Save as CSV for supplementary material
    csv_path = os.path.join(output_dir, "supplementary-results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote CSV supplementary results to {csv_path}")

    # Generate LaTeX longtable
    def fmt_val(v, is_time=False, is_gap=False):
        if v == '-' or pd.isna(v):
            return '-'
        if is_time:
            if v >= timelimit:
                return 'TL'
            return f"{v:.1f}"
        if is_gap:
            if v == 0:
                return '0'
            return f"{v:.1f}"
        if isinstance(v, float):
            if v == int(v):
                return str(int(v))
            return f"{v:.1f}"
        return str(v)

    latex = """\\begin{longtable}{l r r | r r r r r r c | r r r r r c}
\\caption{Per-instance results comparing \\textsc{bnp-full} (branch-price-and-cut) with \\textsc{bnc-sec} (branch-and-cut with subtour elimination).
BKS: best known solution; PB: primal bound; DB: dual bound; DBr: dual bound at root; Gap: optimality gap (\\%); Time: solving time in seconds; Nodes: branch-and-bound nodes; Status: opt (optimal), TL (time limit), open (not solved).}
\\label{tab:supplementary}\\\\
\\toprule
& & & \\multicolumn{7}{c|}{\\textsc{bnp-full}} & \\multicolumn{6}{c}{\\textsc{bnc-sec}} \\\\
Instance & $n$ & BKS & PB & DBr & DB & Gap & Time & Nodes & St. & PB & DB & Gap & Time & Nodes & St. \\\\
\\midrule
\\endfirsthead
\\multicolumn{16}{c}{\\tablename\\ \\thetable{} -- continued from previous page} \\\\
\\toprule
& & & \\multicolumn{7}{c|}{\\textsc{bnp-full}} & \\multicolumn{6}{c}{\\textsc{bnc-sec}} \\\\
Instance & $n$ & BKS & PB & DBr & DB & Gap & Time & Nodes & St. & PB & DB & Gap & Time & Nodes & St. \\\\
\\midrule
\\endhead
\\midrule
\\multicolumn{16}{r}{Continued on next page} \\\\
\\endfoot
\\bottomrule
\\endlastfoot
"""

    for _, row in df.iterrows():
        inst = row['Instance'].replace('_', '\\_')
        n = row['n'] if not pd.isna(row['n']) else '-'
        bks = fmt_val(row['BKS'])

        bnp_pb = fmt_val(row['BNP_PB'])
        bnp_dbr = fmt_val(row['BNP_DB_root'])
        bnp_db = fmt_val(row['BNP_DB'])
        bnp_gap = fmt_val(row['BNP_Gap'], is_gap=True)
        bnp_time = fmt_val(row['BNP_Time'], is_time=True)
        bnp_nodes = fmt_val(row['BNP_Nodes'])
        bnp_st = row['BNP_Status']

        sec_pb = fmt_val(row['SEC_PB'])
        sec_db = fmt_val(row['SEC_DB'])
        sec_gap = fmt_val(row['SEC_Gap'], is_gap=True)
        sec_time = fmt_val(row['SEC_Time'], is_time=True)
        sec_nodes = fmt_val(row['SEC_Nodes'])
        sec_st = row['SEC_Status']

        latex += f"\\texttt{{{inst}}} & {n} & {bks} & {bnp_pb} & {bnp_dbr} & {bnp_db} & {bnp_gap} & {bnp_time} & {bnp_nodes} & {bnp_st} & {sec_pb} & {sec_db} & {sec_gap} & {sec_time} & {sec_nodes} & {sec_st} \\\\\n"

    latex += "\\end{longtable}\n"

    latex_path = os.path.join(output_dir, "supplementary-results.tex")
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"Wrote LaTeX supplementary results to {latex_path}")

    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total instances: {len(df)}")
    bnp_solved = (df['BNP_Status'] == 'opt').sum()
    sec_solved = (df['SEC_Status'] == 'opt').sum()
    print(f"  BNP solved: {bnp_solved}")
    print(f"  SEC solved: {sec_solved}")
    both_solved = ((df['BNP_Status'] == 'opt') & (df['SEC_Status'] == 'opt')).sum()
    only_bnp = ((df['BNP_Status'] == 'opt') & (df['SEC_Status'] != 'opt')).sum()
    only_sec = ((df['BNP_Status'] != 'opt') & (df['SEC_Status'] == 'opt')).sum()
    print(f"  Both solved: {both_solved}")
    print(f"  Only BNP: {only_bnp}")
    print(f"  Only SEC: {only_sec}")

    return df

def branching_comparison(edge_file, rf_file, output_dir="~/papers/lccp-paper/EJOR Version"):
    """
    Compare edge branching vs Ryan-Foster branching.
    Generates LaTeX table for the paper.
    """
    output_dir = os.path.expanduser(output_dir)
    timelimit = 2 * 3600

    edge_df = read_csv(edge_file)
    rf_df = read_csv(rf_file)

    # Get instances solved by at least one
    edge_solved = set(edge_df[(edge_df["final_gap"] == 0) & (edge_df["total_time"] < timelimit)]["instance_name"])
    rf_solved = set(rf_df[(rf_df["final_gap"] == 0) & (rf_df["total_time"] < timelimit)]["instance_name"])

    all_solved = edge_solved | rf_solved
    both_solved = edge_solved & rf_solved
    only_edge = edge_solved - rf_solved
    only_rf = rf_solved - edge_solved

    print(f"Branching Strategy Comparison")
    print(f"=" * 50)
    print(f"Edge branching solved: {len(edge_solved)}")
    print(f"Ryan-Foster solved: {len(rf_solved)}")
    print(f"Both solved: {len(both_solved)}")
    print(f"Only edge: {len(only_edge)}")
    print(f"Only RF: {len(only_rf)}")

    # Time comparison on commonly solved instances
    if len(both_solved) > 0:
        edge_times = edge_df[edge_df["instance_name"].isin(both_solved)]["total_time"]
        rf_times = rf_df[rf_df["instance_name"].isin(both_solved)]["total_time"]
        print(f"\nOn {len(both_solved)} commonly solved instances:")
        print(f"  Edge SGM time: {sgm(edge_times, 1):.2f}s")
        print(f"  RF SGM time: {sgm(rf_times, 1):.2f}s")

        # Node comparison
        edge_nodes = edge_df[edge_df["instance_name"].isin(both_solved)]["number_of_nodes"]
        rf_nodes = rf_df[rf_df["instance_name"].isin(both_solved)]["number_of_nodes"]
        print(f"  Edge avg nodes: {edge_nodes.mean():.1f}")
        print(f"  RF avg nodes: {rf_nodes.mean():.1f}")

    # Generate LaTeX table
    latex = """\\small
\\begin{tabular}{lrrrr}
    \\toprule
    Branching & Solved & Time [s] & Nodes & Ratio \\\\
    \\midrule
"""
    edge_time = sgm(edge_df[edge_df["instance_name"].isin(all_solved)]["total_time"], 1)
    rf_time = sgm(rf_df[rf_df["instance_name"].isin(all_solved)]["total_time"], 1)
    edge_nodes_avg = edge_df[edge_df["instance_name"].isin(all_solved)]["number_of_nodes"].mean()
    rf_nodes_avg = rf_df[rf_df["instance_name"].isin(all_solved)]["number_of_nodes"].mean()

    latex += f"    Edge & {len(edge_solved)} & {edge_time:.1f} & {edge_nodes_avg:.1f} & 1.00 \\\\\n"
    latex += f"    Ryan-Foster & {len(rf_solved)} & {rf_time:.1f} & {rf_nodes_avg:.1f} & {rf_time/edge_time:.2f} \\\\\n"
    latex += "    \\bottomrule\n\\end{tabular}\n"

    latex_path = os.path.join(output_dir, "branching-comparison.tex")
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"\nWrote branching comparison to {latex_path}")


def dual_stabilization_comparison(baseline_file, stabilized_file, output_dir="~/papers/lccp-paper/EJOR Version"):
    """
    Compare baseline vs dual stabilization.
    """
    output_dir = os.path.expanduser(output_dir)
    timelimit = 2 * 3600

    base_df = read_csv(baseline_file)
    stab_df = read_csv(stabilized_file)

    base_solved = set(base_df[(base_df["final_gap"] == 0) & (base_df["total_time"] < timelimit)]["instance_name"])
    stab_solved = set(stab_df[(stab_df["final_gap"] == 0) & (stab_df["total_time"] < timelimit)]["instance_name"])

    all_solved = base_solved | stab_solved
    both_solved = base_solved & stab_solved

    print(f"Dual Stabilization Comparison")
    print(f"=" * 50)
    print(f"Baseline solved: {len(base_solved)}")
    print(f"Stabilized solved: {len(stab_solved)}")
    print(f"Both solved: {len(both_solved)}")

    if len(both_solved) > 0:
        base_times = base_df[base_df["instance_name"].isin(both_solved)]["total_time"]
        stab_times = stab_df[stab_df["instance_name"].isin(both_solved)]["total_time"]
        print(f"\nOn {len(both_solved)} commonly solved instances:")
        print(f"  Baseline SGM time: {sgm(base_times, 1):.2f}s")
        print(f"  Stabilized SGM time: {sgm(stab_times, 1):.2f}s")
        print(f"  Speedup: {sgm(base_times, 1) / sgm(stab_times, 1):.2f}x")

        # CG iterations comparison if available
        if "cg_iterations_total" in base_df.columns and "cg_iterations_total" in stab_df.columns:
            base_cg = base_df[base_df["instance_name"].isin(both_solved)]["cg_iterations_total"]
            stab_cg = stab_df[stab_df["instance_name"].isin(both_solved)]["cg_iterations_total"]
            print(f"  Baseline avg CG iters: {base_cg.mean():.1f}")
            print(f"  Stabilized avg CG iters: {stab_cg.mean():.1f}")


def cg_iterations_analysis(csv_files):
    """
    Analyze column generation iterations across solvers.
    """
    print(f"CG Iterations Analysis")
    print(f"=" * 50)

    for f in csv_files:
        df = read_csv(f)
        solver_name = f.split("/")[-1].replace(".csv", "")

        if "cg_iterations_total" in df.columns:
            total_cg = df["cg_iterations_total"].mean()
            root_cg = df["cg_iterations_root"].mean() if "cg_iterations_root" in df.columns else np.nan
            print(f"{solver_name:30s} | Total CG: {total_cg:8.1f} | Root CG: {root_cg:8.1f}")
        else:
            print(f"{solver_name:30s} | No CG iteration data")


def greedy_heuristic_analysis(results_dir, output_dir="~/papers/lccp-paper/EJOR Version"):
    """
    Analyze greedy heuristic performance by parsing solver logs.
    Extracts: success rate, columns added, reduced costs, etc.
    """
    import re
    output_dir = os.path.expanduser(output_dir)

    sol_files = glob.glob(os.path.join(results_dir, "*.sol"))
    if not sol_files:
        print(f"No .sol files found in {results_dir}")
        return

    all_stats = []

    for sol_file in sol_files:
        instance = os.path.basename(sol_file).replace(".json.sol", "")

        with open(sol_file, 'r') as f:
            content = f.read()

        # Count greedy successes and failures
        greedy_success = len(re.findall(r'> heuristically found variables: (\d+)', content))
        greedy_fail = len(re.findall(r'> no heuristically found variables', content))
        total_pricing = greedy_success + greedy_fail

        # Extract columns found by greedy
        greedy_cols = re.findall(r'> heuristically found variables: (\d+)', content)
        greedy_cols = [int(x) for x in greedy_cols]
        total_greedy_cols = sum(greedy_cols)

        # Extract best reduced costs from greedy
        greedy_rc = re.findall(r'best reduced cost: (-?[\d.e+-]+)', content)
        greedy_rc = [float(x) for x in greedy_rc]
        avg_greedy_rc = np.mean(greedy_rc) if greedy_rc else np.nan

        # Extract labeling columns (added paths after no greedy)
        labeling_cols = re.findall(r'> added paths: (\d+)', content)
        labeling_cols = [int(x) for x in labeling_cols]
        total_labeling_cols = sum(labeling_cols)

        # Neighborhood expansion stats
        neighborhood_pcts = re.findall(r'larger neighborhood \((\d+)%\)', content)
        neighborhood_pcts = [int(x) for x in neighborhood_pcts]
        max_neighborhood = max(neighborhood_pcts) if neighborhood_pcts else 0
        ng_expansions = len(neighborhood_pcts)

        all_stats.append({
            'instance': instance,
            'greedy_success': greedy_success,
            'greedy_fail': greedy_fail,
            'total_pricing': total_pricing,
            'success_rate': greedy_success / total_pricing * 100 if total_pricing > 0 else 0,
            'greedy_cols': total_greedy_cols,
            'labeling_cols': total_labeling_cols,
            'avg_greedy_rc': avg_greedy_rc,
            'ng_expansions': ng_expansions,
            'max_neighborhood': max_neighborhood,
        })

    df = pd.DataFrame(all_stats)

    print(f"Greedy Heuristic Analysis ({results_dir})")
    print(f"=" * 60)
    print(f"Instances analyzed: {len(df)}")
    print(f"")
    print(f"Success Rate:")
    print(f"  Mean: {df['success_rate'].mean():.1f}%")
    print(f"  Min:  {df['success_rate'].min():.1f}%")
    print(f"  Max:  {df['success_rate'].max():.1f}%")
    print(f"")
    print(f"Columns Added:")
    print(f"  Total greedy:   {df['greedy_cols'].sum():,}")
    print(f"  Total labeling: {df['labeling_cols'].sum():,}")
    print(f"  Greedy ratio:   {df['greedy_cols'].sum() / (df['greedy_cols'].sum() + df['labeling_cols'].sum()) * 100:.1f}%")
    print(f"")
    print(f"Average Greedy Reduced Cost: {df['avg_greedy_rc'].mean():.4f}")
    print(f"")
    print(f"NG-Relaxation:")
    print(f"  Instances with expansions: {(df['ng_expansions'] > 0).sum()}")
    print(f"  Avg expansions per instance: {df['ng_expansions'].mean():.1f}")
    print(f"  Max neighborhood reached: {df['max_neighborhood'].max()}%")

    # Save detailed results
    csv_path = os.path.join(output_dir, "greedy-analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote detailed results to {csv_path}")

    return df


def best_known_sols(df):
    bks = {}
    for instance in df.instance_name.unique():
        bks[instance] = df[df.instance_name == instance].primal_bound.min()
    return bks

def get_best_known_solutions(csv_files):
    """
    Creates a dictionary of best known solutions from multiple CSV result files.
    Returns a dictionary mapping instance names to tuples of (best_value, is_proven_optimal).
    Prints the results as JSON to stdout.
    """
    import json
    best_solutions = {}
    for file in csv_files:
        df = read_csv(file)
        for _, row in df.iterrows():
            instance = row["instance_name"]
            value = row["primal_bound"]
            is_optimal = row["final_gap"] == 0 and row["total_time"] < 10600
            
            if instance not in best_solutions:
                best_solutions[instance] = (value, is_optimal)
            else:
                current_value, current_optimal = best_solutions[instance]
                # Update if we found a better solution or if this one is optimal
                if value < current_value or (value == current_value and is_optimal):
                    best_solutions[instance] = (value, is_optimal)
    
    # Convert tuples to dictionaries for JSON serialization
    json_output = {
        instance: {
            "best_value": value,
            "is_proven_optimal": is_optimal
        }
        for instance, (value, is_optimal) in best_solutions.items()
    }
    with open("best_known_solutions.json", "w") as f:
        json.dump(json_output, f, indent=4)
    return best_solutions

def validate_against_best_known(csv_file, best_known_file="best_known_solutions.json"):
    """
    Validates results in a CSV file against the best known solutions.
    Checks if the solutions:
    - Match or are better/worse than best known values
    - Falsely claim optimality for suboptimal solutions
    - Find new optimality proofs
    - Have dual bounds that violate best known primal bounds
    
    Args:
        csv_file: Path to the CSV file to validate
        best_known_file: Path to the JSON file containing best known solutions
    """
    import json
    
    # Load best known solutions
    with open(best_known_file, 'r') as f:
        best_known = json.load(f)
    
    # Read the CSV file
    df = read_csv(csv_file)
    
    print(f"\nValidating {csv_file} against best known solutions:")
    print("-" * 80)
    
    differences = []
    for _, row in df.iterrows():
        instance = row["instance_name"]
        if instance in best_known:
            current_val = row["primal_bound"]
            best_val = best_known[instance]["best_value"]
            is_known_optimal = best_known[instance]["is_proven_optimal"]
            claims_optimal = row["final_gap"] == 0 and row["total_time"] < 10600
            
            # Check if dual bound violates best known primal bound
            dual_bound = row["dual_bound"]
            dual_bound_root = row["dual_bound_at_root"]
            if dual_bound > best_val or dual_bound_root > best_val:
                differences.append(f"ERROR: Invalid dual bound for {instance} - dual bound {max(dual_bound, dual_bound_root)} exceeds best known value {best_val}")
            
            if current_val < best_val:
                differences.append(f"Better solution found for {instance}: {current_val} (previous best: {best_val})")
            elif current_val > best_val:
                differences.append(f"Worse solution for {instance}: {current_val} (best known: {best_val})")
            
            # Check for false optimality claims
            if claims_optimal:
                if current_val > best_val:
                    differences.append(f"WARNING: False optimality claim for {instance} - claims {current_val} is optimal but better solution exists: {best_val}")
                elif not is_known_optimal:
                    differences.append(f"New optimality proof for {instance}")
            
            # Check for suboptimal solutions claimed as optimal
            if claims_optimal and not is_known_optimal and current_val > best_val:
                differences.append(f"WARNING: Suboptimal solution claimed as optimal for {instance}: got {current_val}, best known is {best_val}")
        else:
            differences.append(f"New instance found: {instance} with value {row['primal_bound']}")
    
    if differences:
        print("Differences found:")
        for diff in differences:
            print(f"- {diff}")
    else:
        print("All solutions match best known values")
    print("-" * 80)
    return len(differences) == 0

def diff_obj_val(file1, file2): 
    print(file1, file2)
    df1 = read_csv(file1)
    df2 = read_csv(file2)
    solved1 = df1[(df1["final_gap"] == 0) & (df1["total_time"] < timelimit + 200)]["instance_name"].unique()
    solved2 = df2[(df2["final_gap"] == 0) & (df2["total_time"] < timelimit + 20)]["instance_name"].unique()
    solved_by_both = set(solved1) & set(solved2)
    # df1 = df1[df1["instance_name"].isin(solved_by_both)]
    # df2 = df2[df2["instance_name"].isin(solved_by_both)]

    bks1 = best_known_sols(df1)
    bks2 = best_known_sols(df2)

    any_diff = False
    for instance in bks1:
        if instance not in bks2:
            print(f"instance {instance} not in bks2")
            continue
        diff = bks1[instance] - bks2[instance]
        if diff != 0:
            any_diff = True
            print(instance, bks1[instance], bks2[instance], bks1[instance] - bks2[instance])

    if not any_diff:
        print("no differences")
        print(len(solved1), len(solved2), len(solved_by_both))
        print(sum(bks1.values()))
        print(sum(bks2.values()))

def read_csv(path):
    df = pd.read_csv(path)
    # if df["instance_name"][0][:3] in ["t84", "at8", "at6", "t62"]:
    #     df = df[df.instance_name.str.startswith("t84_")]
    #     df["instance_name"] = df.instance_name.map(lambda x: x.split("_")[-1])
    df["total_time"] = df["total_time"].map(fix_time)
    if "compact" not in path:
        # True optimality gap (uses the FINAL dual bound) -> determines solved status.
        df["final_gap"] = (df["primal_bound"] - df["dual_bound"]) / df["primal_bound"]
        # Root gap (uses the root dual bound) -> for root-bound / "at root" metrics only.
        df["root_gap"] = (df["primal_bound"] - df["dual_bound_at_root"]) / df["primal_bound"]
    return df


def fix_time(time):
    try:
        number = float(time)
        if math.isnan(number):
            return 7230.
        return number
    except:
        return 7230.

def collect_metrics_for_file(file, timelimit):
    """Process a single file and return metrics for the first pass."""
    df = read_csv(file)
    solved = set(df[(df["final_gap"] == 0) & (df["total_time"] < timelimit + 20)]["instance_name"].unique())
    solved_at_root = df[(df["final_gap"] == 0) & (df["total_time"] < timelimit + 20) & (df["number_of_nodes"] == 1)]["instance_name"].unique()
    nrootsolved = df[(df["dual_bound_at_root"] > 1) | ((df["final_gap"] == 0) & (df["total_time"] < timelimit + 20))]["instance_name"].unique()

    if "first_primal_bound" in df.columns:
        optimalprimalatroot = set(df[(df["first_primal_bound"] == df["primal_bound"])]["instance_name"].unique()) & solved
    else:
        optimalprimalatroot = set(df[(df["primal_bound_at_root"] == df["primal_bound"])]["instance_name"].unique()) & solved

    raised_dualbound = df[df["dual_bound"] > df["dual_bound_at_root"]]["instance_name"].unique()
    if "first_primal_bound" in df.columns:
        dropped_primalbound = df[df["primal_bound"] < df["first_primal_bound"]]["instance_name"].unique()
    else:
        dropped_primalbound = df[df["primal_bound"] < df["primal_bound_at_root"]]["instance_name"].unique()

    solver_name = file.split("/")[-1].replace(".csv", "")
    all_instances = set(df["instance_name"].unique())

    return {
        "file": file,
        "solver_name": solver_name,
        "solved": solved,
        "solved_at_root_count": len(solved_at_root),
        "root_solved_count": len(nrootsolved),
        "optimal_primal_count": len(optimalprimalatroot),
        "raised_dual_count": len(raised_dualbound),
        "dropped_primal_count": len(dropped_primalbound),
        "all_instances": all_instances,
    }

def compute_stats_for_file(file, timelimit, instances_solved_by_any, instances_solved_by_all, file_name_map):
    """Process a single file and return statistics for the second pass."""
    file_name = file.split("/")[-1].replace(".csv", "")

    df = pd.read_csv(file)
    df["total_time"] = df["total_time"].map(fix_time)
    if "compact" not in file:
        # True optimality gap (FINAL dual bound) -> solved status & avg gap.
        df["final_gap"] = (df["primal_bound"] - df["dual_bound"]) / df["primal_bound"]
        # Root gap (root dual bound) -> root-bound metrics only.
        df["root_gap"] = (df["primal_bound"] - df["dual_bound_at_root"]) / df["primal_bound"]

    optimal = df[(df['final_gap'] == 0.0) & (df["total_time"] < timelimit + 30)]
    optimal_count = optimal.shape[0]

    df_any = df[df["instance_name"].isin(instances_solved_by_any)]
    df_all = df[df["instance_name"].isin(instances_solved_by_all)]

    df_any = df_any.copy()
    df_any["solver"] = file_name_map.get(file, file_name)

    avg_time = sgm(df_any['total_time'], 1)
    avg_node = df_any['number_of_nodes'].mean()
    avg_time_all = sgm(df_all['total_time'], 1)
    avg_node_all = df_all['number_of_nodes'].mean()

    # CG iterations (handle missing columns gracefully)
    avg_cg_total = df_any['cg_iterations_total'].mean() if 'cg_iterations_total' in df_any.columns else np.nan
    avg_cg_root = df_any['cg_iterations_root'].mean() if 'cg_iterations_root' in df_any.columns else np.nan
    avg_cg_total_all = df_all['cg_iterations_total'].mean() if 'cg_iterations_total' in df_all.columns else np.nan
    avg_cg_root_all = df_all['cg_iterations_root'].mean() if 'cg_iterations_root' in df_all.columns else np.nan

    return {
        "file": file,
        "file_name": file_name,
        "optimal_count": optimal_count,
        "df_any": df_any,
        "avg_time": avg_time,
        "avg_node": avg_node,
        "avg_time_all": avg_time_all,
        "avg_node_all": avg_node_all,
        "avg_gap": df_any["final_gap"].mean(),
        "avg_obj": df_any["primal_bound"].mean(),
        "total_obj": df_any["primal_bound"].sum(),
        "avg_cg_total": avg_cg_total,
        "avg_cg_root": avg_cg_root,
        "avg_cg_total_all": avg_cg_total_all,
        "avg_cg_root_all": avg_cg_root_all,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("patterns", nargs="*", default=None,
                        help="Glob patterns or file paths (e.g., 'results/25_*.csv' or 'file1.csv file2.csv')")
    parser.add_argument("--validate", "-v", action="store_true",
                        help="Validate results against best known solutions")
    parser.add_argument("--supplementary", "-s", action="store_true",
                        help="Generate per-instance supplementary material tables")
    parser.add_argument("--branching", nargs=2, metavar=("EDGE_FILE", "RF_FILE"),
                        help="Compare edge vs Ryan-Foster branching")
    parser.add_argument("--stabilization", nargs=2, metavar=("BASE_FILE", "STAB_FILE"),
                        help="Compare baseline vs dual stabilization")
    parser.add_argument("--cg-analysis", action="store_true",
                        help="Analyze CG iterations across loaded files")
    parser.add_argument("--root-bounds", action="store_true",
                        help="Compare root bounds between BNP and SEC")
    parser.add_argument("--greedy", metavar="RESULTS_DIR",
                        help="Analyze greedy heuristic from log files in directory")
    args = parser.parse_args()

    if args.supplementary:
        generate_supplementary_tables()
        exit(0)

    if args.branching:
        branching_comparison(args.branching[0], args.branching[1])
        exit(0)

    if args.stabilization:
        dual_stabilization_comparison(args.stabilization[0], args.stabilization[1])
        exit(0)

    if args.root_bounds:
        root_bound_comparison()
        exit(0)

    if args.greedy:
        greedy_heuristic_analysis(args.greedy)
        exit(0)

    if args.patterns:
        csv_files = []
        for pattern in args.patterns:
            matched = glob.glob(pattern)
            if not matched:
                print(f"Warning: No files matched pattern: {pattern}")
            csv_files.extend(matched)
        csv_files = sorted(set(csv_files))  # Remove duplicates and sort
        if not csv_files:
            print("No files matched any of the provided patterns")
            exit(1)
        # Always include compact.csv as baseline
        compact_path = "results/compact.csv"
        if os.path.exists(compact_path) and compact_path not in csv_files:
            csv_files.insert(0, compact_path)
    else:
        # Automatically load the latest experiment files
        csv_files = get_latest_experiment_files()
    print(f"Loading experiment files: {csv_files}")


    # Initialize lists to hold the results for each file
    optimal_counts = []
    avg_times = []
    avg_nodes = []
    avg_gap = []
    avg_obj = []
    total_obj = []
    solved_count = {
        # 1: [],
        # 10: [],
        # 100: [],
        # 1000: [],
    }

    for i in range(1, 10600, 1):
        solved_count[i] = []

    file_name_map = {
        "results/compact.csv": "compact",
        # "results/bnp0910.csv": "bnp",
        # "results/bnp0910_nopar.csv": "bnp-nopar",
        # "results/bnp0910_nobidir.csv": "bnp-nobidir",
        # "results/bnp0910_nosymbr.csv": "bnp-nosymbr",
        # "results/bnp0910_noint.csv": "bnp-noint",
        # "results/bnp0910_trieq.csv": "bnp-trieq",
        # "results/bnp0910_trieq-opt2.csv": "bnp-trieq-opt",
        # "results/bnp0412.csv": "bnp-dom",
        # "results/bnp2701.csv": "bnp",
        # "results/bnp2701_nopar.csv": "bnp-nopar",
        # "results/bnp2701_nobidir.csv": "bnp-nobidir",
        # "results/bnp2701_nosymbr.csv": "bnp-nosymbr",
        # "results/bnp2701_noint.csv": "bnp-noint",
        # "results/bnp2701_trieq.csv": "bnp-trieq",
        # "results/bnp2701_trieq-opt.csv": "bnp-trieq-opt",
        # "results/bnp2701_basic.csv": "bnp-basic",
    }


    big_df = pd.DataFrame()
    instances_solved_by_all = set()
    instances_solved_by_any = set()
    all_instances = set()
    timelimit = 2 * 3600

    # Additional metrics to collect
    solved_at_root_counts = []
    root_solved_counts = []
    primal_heur_optimal_counts = []
    raised_dual_counts = []
    improved_primal_counts = []
    # Metrics over solved_by_all
    avg_times_all = []
    avg_nodes_all = []
    # CG iteration metrics
    avg_cg_totals = []
    avg_cg_roots = []
    avg_cg_totals_all = []
    avg_cg_roots_all = []
    # Track solved instances per solver
    solved_by_solver = {}

    # First pass: collect metrics in parallel
    print("Collecting metrics (parallel)...")
    first_pass_results = {}
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(collect_metrics_for_file, f, timelimit): f for f in csv_files}
        for future in tqdm(as_completed(futures), total=len(csv_files), desc="Collecting metrics"):
            result = future.result()
            first_pass_results[result["file"]] = result

    # Aggregate results from first pass (must be done sequentially for set operations)
    for file in csv_files:
        result = first_pass_results[file]
        solved_at_root_counts.append(result["solved_at_root_count"])
        root_solved_counts.append(result["root_solved_count"])
        primal_heur_optimal_counts.append(result["optimal_primal_count"])
        raised_dual_counts.append(result["raised_dual_count"])
        improved_primal_counts.append(result["dropped_primal_count"])
        instances_solved_by_any = instances_solved_by_any.union(result["solved"])
        instances_solved_by_all = instances_solved_by_all.intersection(result["solved"]) if len(instances_solved_by_all) else result["solved"]
        all_instances = all_instances.union(result["all_instances"])
        solved_by_solver[result["solver_name"]] = result["solved"]

    # Second pass: compute statistics in parallel
    print("\n" + "-"*80)
    print("Processing results (parallel):")
    print("-"*80)
    second_pass_results = {}
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(compute_stats_for_file, f, timelimit, instances_solved_by_any, instances_solved_by_all, file_name_map): f for f in csv_files}
        for future in tqdm(as_completed(futures), total=len(csv_files), desc="Computing statistics"):
            result = future.result()
            second_pass_results[result["file"]] = result

    # Aggregate results from second pass
    for file in csv_files:
        result = second_pass_results[file]
        optimal_counts.append(result["optimal_count"])
        avg_times.append(result["avg_time"])
        avg_nodes.append(result["avg_node"])
        avg_times_all.append(result["avg_time_all"])
        avg_nodes_all.append(result["avg_node_all"])
        avg_gap.append(result["avg_gap"])
        avg_obj.append(result["avg_obj"])
        total_obj.append(result["total_obj"])
        avg_cg_totals.append(result["avg_cg_total"])
        avg_cg_roots.append(result["avg_cg_root"])
        avg_cg_totals_all.append(result["avg_cg_total_all"])
        avg_cg_roots_all.append(result["avg_cg_root_all"])
        big_df = pd.concat([big_df, result["df_any"]])
        print(f"  {result['file_name']:25s} | Solved: {result['optimal_count']:3d} | Time(SGM): {result['avg_time']:8.2f} | Nodes: {result['avg_node']:10.2f} | Gap: {result['avg_gap']:6.2%}")
    
    print("instances", len(instances_solved_by_all))

    # Compute virtual best and virtual worst
    vb_times = big_df.groupby("instance_name")["total_time"].min()
    vw_times = big_df.groupby("instance_name")["total_time"].max()
    vb_nodes = big_df.groupby("instance_name")["number_of_nodes"].min()
    vw_nodes = big_df.groupby("instance_name")["number_of_nodes"].max()

    # Filter to solved_by_all for the "All" metrics
    vb_times_all = big_df[big_df["instance_name"].isin(instances_solved_by_all)].groupby("instance_name")["total_time"].min()
    vw_times_all = big_df[big_df["instance_name"].isin(instances_solved_by_all)].groupby("instance_name")["total_time"].max()
    vb_nodes_all = big_df[big_df["instance_name"].isin(instances_solved_by_all)].groupby("instance_name")["number_of_nodes"].min()
    vw_nodes_all = big_df[big_df["instance_name"].isin(instances_solved_by_all)].groupby("instance_name")["number_of_nodes"].max()

    # Compute unique solves and solves not in {N}_full
    file_names = [s.split("/")[-1].replace(".csv","") for s in csv_files]
    unique_solves = []
    not_in_full = []
    # Find the full solver matching pattern {number}_full
    full_solver_name = None
    for name in solved_by_solver.keys():
        if re.match(r'^\d+_full$', name):
            full_solver_name = name
            break
    full_solved = solved_by_solver.get(full_solver_name, set()) if full_solver_name else set()
    for solver_name in file_names:
        solver_solved = solved_by_solver.get(solver_name, set())
        # Instances only solved by this solver
        others_solved = set()
        for other_name, other_solved in solved_by_solver.items():
            if other_name != solver_name:
                others_solved = others_solved.union(other_solved)
        unique = solver_solved - others_solved
        unique_solves.append(len(unique))
        # Instances solved by this solver but not by {N}_full
        not_in_full.append(len(solver_solved - full_solved))

    # Compute time ratio compared to full solver
    full_solver_idx = None
    if full_solver_name:
        for i, name in enumerate(file_names):
            if name == full_solver_name:
                full_solver_idx = i
                break
    full_time = avg_times[full_solver_idx] if full_solver_idx is not None else None
    time_ratios = [t / full_time if full_time else np.nan for t in avg_times]

    # Compute wins over full (instances solved faster by more than 0.5 seconds)
    wins_over_full = []
    if full_solver_name:
        full_times_df = big_df[big_df["solver"] == full_solver_name][["instance_name", "total_time"]]
        full_times_dict = dict(zip(full_times_df["instance_name"], full_times_df["total_time"]))
        for solver_name in file_names:
            solver_df = big_df[big_df["solver"] == solver_name]
            wins = 0
            for _, row in solver_df.iterrows():
                instance = row["instance_name"]
                if instance in full_times_dict:
                    full_t = full_times_dict[instance]
                    solver_t = row["total_time"]
                    if solver_t < full_t * 0.9:  # solver is faster by more than 10%
                        wins += 1
            wins_over_full.append(wins)
    else:
        wins_over_full = [np.nan] * len(file_names)

    # Create summary table
    summary_df = pd.DataFrame({
        "Solver": file_names + ["VirtualBest", "VirtualWorst"],
        "Solved": optimal_counts + [len(instances_solved_by_any), len(instances_solved_by_all)],
        "Unique": unique_solves + [0, 0],
        "!Full": not_in_full + [0, 0],
        "AtRoot": solved_at_root_counts + [np.nan, np.nan],
        "RootDB>1": root_solved_counts + [np.nan, np.nan],
        "HeurOpt": primal_heur_optimal_counts + [np.nan, np.nan],
        "RaisedDB": raised_dual_counts + [np.nan, np.nan],
        "ImprPB": improved_primal_counts + [np.nan, np.nan],
        "Time(SGM)": avg_times + [sgm(vb_times, 1), sgm(vw_times, 1)],
        "T/Full": time_ratios + [sgm(vb_times, 1) / full_time if full_time else np.nan, sgm(vw_times, 1) / full_time if full_time else np.nan],
        "Wins": wins_over_full + [np.nan, np.nan],
        "Nodes": avg_nodes + [vb_nodes.mean(), vw_nodes.mean()],
        "Gap": avg_gap + [np.nan, np.nan],
        "CG(Tot)": avg_cg_totals + [np.nan, np.nan],
        "CG(Root)": avg_cg_roots + [np.nan, np.nan],
        "Time(All)": avg_times_all + [sgm(vb_times_all, 1), sgm(vw_times_all, 1)],
        "Nodes(All)": avg_nodes_all + [vb_nodes_all.mean(), vw_nodes_all.mean()],
        "CG(Tot,All)": avg_cg_totals_all + [np.nan, np.nan],
        "CG(Root,All)": avg_cg_roots_all + [np.nan, np.nan],
    })
    # Format for display
    display_df = summary_df.copy()
    int_cols = ["Solved", "Unique", "!Full", "AtRoot", "RootDB>1", "HeurOpt", "RaisedDB", "ImprPB", "Wins"]
    float_cols = ["Time(SGM)", "T/Full", "Nodes", "Gap", "CG(Tot)", "CG(Root)", "Time(All)", "Nodes(All)", "CG(Tot,All)", "CG(Root,All)"]
    for col in int_cols:
        display_df[col] = display_df[col].apply(lambda x: "-" if pd.isna(x) else f"{int(x)}")
    for col in float_cols:
        display_df[col] = display_df[col].apply(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
    table_str = display_df.to_string(index=False)
    table_width = len(table_str.split("\n")[0])
    print("\n" + "="*table_width)
    print("SUMMARY TABLE".center(table_width))
    print("="*table_width)
    print(table_str)
    print("="*table_width + "\n")

    best_known_sols = json.load(open("best_known_solutions.json"))

    # Generate LaTeX tables
    paper_dir = os.path.expanduser("~/papers/lccp-paper/EJOR Version")

    # Table 1: Aggregated performance comparison (ablation study)
    # Filter out unsafe variants and VirtualBest/VirtualWorst for ablation table
    ablation_df = summary_df[~summary_df["Solver"].str.contains("unsafe|Virtual|srow|clique", case=False, regex=True)].copy()

    # Create nice solver names
    solver_name_map = {
        "compact": "\\textsc{bnc-sec}",
        "27_full": "\\textsc{bnp-full}",
        "27_full_nobidir": "\\textsc{bnp-nobidir}",
        "27_full_nopar": "\\textsc{bnp-nopar}",
        "27_full_nosymbr": "\\textsc{bnp-nosymbr}",
        "27_full_noearly": "\\textsc{bnp-noearly}",
        "27_full_nogreedy": "\\textsc{bnp-nogreedy}",
        "27_full_trieq": "\\textsc{bnp-trieq}",
        "27_full_nobestsol": "\\textsc{bnp-nobestsol}",
    }
    ablation_df["Solver"] = ablation_df["Solver"].map(lambda x: solver_name_map.get(x, x))
    ablation_df = ablation_df[["Solver", "Solved", "Time(SGM)", "T/Full", "Gap"]]
    ablation_df.columns = ["Solver", "Solved", "Time [s]", "Ratio", "Gap"]

    # Format the table
    ablation_latex = "\\small\n\\begin{tabular}{lrrrr}\n    \\toprule\n"
    ablation_latex += "    Solver & Solved & Time [s] & Ratio & Gap (\\%) \\\\\n"
    ablation_latex += "    \\midrule\n"

    for _, row in ablation_df.iterrows():
        solver = row["Solver"]
        solved = int(row["Solved"])
        time_val = row["Time [s]"]
        ratio = row["Ratio"]
        gap = row["Gap"] * 100  # Convert to percentage

        # Bold the best (full) solver
        if "bnp-full" in solver:
            ablation_latex += f"    {solver} & \\textbf{{{solved}}} & \\textbf{{{time_val:.1f}}} & \\textbf{{{ratio:.2f}}} & \\textbf{{{gap:.1f}}} \\\\\n"
            ablation_latex += "    \\midrule\n"
        elif "bnc-sec" in solver:
            ablation_latex += f"    {solver} & {solved} & {time_val:.1f} & {ratio:.2f} & {gap:.1f} \\\\\n"
        else:
            ablation_latex += f"    {solver} & {solved} & {time_val:.1f} & {ratio:.2f} & {gap:.1f} \\\\\n"

    ablation_latex += "    \\bottomrule\n\\end{tabular}\n"

    ablation_path = os.path.join(paper_dir, "aggregated.tex")
    with open(ablation_path, "w") as f:
        f.write(ablation_latex)
    print(f"Wrote aggregated table to {ablation_path}")

    # Table 2: Instance-level results (compact vs 27_full)
    compact_df = pd.read_csv("results/compact.csv")
    full_df = read_csv("results/27_full.csv")

    compact_df.sort_values(by=["instance_name"], inplace=True)
    full_df.sort_values(by=["instance_name"], inplace=True)

    # Merge the two dataframes
    merged_df = full_df.merge(compact_df, on=["instance_name"], suffixes=("_bnp", "_compact"))
    merged_df["input_nodes"] = merged_df["instance_name"].map(lambda x: extract_number_from_filename(x))
    merged_df.sort_values(by="input_nodes", ascending=True, inplace=True)

    # Format times
    def format_time(t, tl=timelimit):
        if pd.isna(t) or t >= tl + 200:
            return "TL"
        return f"{max(0.01, t):.2f}"

    def format_gap(g):
        if pd.isna(g):
            return "-"
        return f"{g * 100:.1f}"

    # Build LaTeX longtable
    instance_latex = "\\begin{longtable}{|l|c|rrrr|rrrrr|}\n"
    instance_latex += "\\toprule\n"
    instance_latex += "Instance & Nodes & \\multicolumn{4}{c|}{\\textsc{bnc-sec}} & \\multicolumn{5}{c|}{\\textsc{bnp-full}} \\\\\n"
    instance_latex += " &  & DB & PB & Gap & Time(s) & DBroot & DB & PB & Gap & Time(s) \\\\\n"
    instance_latex += "\\midrule\n"
    instance_latex += "\\endfirsthead\n"
    instance_latex += "\\toprule\n"
    instance_latex += "Instance & Nodes & \\multicolumn{4}{c|}{\\textsc{bnc-sec}} & \\multicolumn{5}{c|}{\\textsc{bnp-full}} \\\\\n"
    instance_latex += " &  & DB & PB & Gap & Time(s) & DBroot & DB & PB & Gap & Time(s) \\\\\n"
    instance_latex += "\\midrule\n"
    instance_latex += "\\endhead\n"
    instance_latex += "\\midrule\n"
    instance_latex += "\\multicolumn{11}{r}{Continued on next page} \\\\\n"
    instance_latex += "\\midrule\n"
    instance_latex += "\\endfoot\n"
    instance_latex += "\\bottomrule\n"
    instance_latex += "\\endlastfoot\n"

    for _, row in merged_df.iterrows():
        instance_name = row["instance_name"].replace("_", "\\_")
        nodes = row["input_nodes"]

        # Compact columns
        db_compact = row["dual_bound_compact"]
        pb_compact = row["primal_bound_compact"]
        gap_compact = row["final_gap_compact"]
        time_compact = format_time(row["total_time_compact"])

        # BNP columns
        db_root_bnp = row["dual_bound_at_root_bnp"]
        db_bnp = row["dual_bound_bnp"]
        pb_bnp = row["primal_bound_bnp"]
        gap_bnp = row["final_gap_bnp"]
        time_bnp = format_time(row["total_time_bnp"])

        instance_latex += f"\\texttt{{{instance_name}}} & {nodes} & "
        instance_latex += f"{db_compact:.1f} & {pb_compact:.1f} & {format_gap(gap_compact)} & {time_compact} & "
        instance_latex += f"{db_root_bnp:.1f} & {db_bnp:.1f} & {pb_bnp:.1f} & {format_gap(gap_bnp)} & {time_bnp} \\\\\n"

    instance_latex += "\\end{longtable}\n"

    instance_path = os.path.join(paper_dir, "all-instances.tex")
    with open(instance_path, "w") as f:
        f.write(instance_latex)
    print(f"Wrote instance table to {instance_path}")

    # Table 3: Instance analysis (solved, root stats, etc.)
    # Filter out unsafe, cuts, and virtual variants for this table
    analysis_df = summary_df[~summary_df["Solver"].str.contains("unsafe|Virtual|srow|clique", case=False, regex=True)].copy()
    analysis_df["Solver"] = analysis_df["Solver"].map(lambda x: solver_name_map.get(x, x))

    # Select relevant columns for instance analysis
    analysis_df = analysis_df[["Solver", "Solved", "AtRoot", "RootDB>1", "HeurOpt", "ImprPB"]]

    # Build LaTeX table
    analysis_latex = "\\small\n\\begin{tabular}{lrrrrr}\n    \\toprule\n"
    analysis_latex += "    Solver & Solved & At Root & Root LP & Heur Opt & Impr. PB \\\\\n"
    analysis_latex += "    \\midrule\n"

    for _, row in analysis_df.iterrows():
        solver = row["Solver"]
        solved = int(row["Solved"])
        at_root = int(row["AtRoot"]) if not pd.isna(row["AtRoot"]) else "-"
        root_lp = int(row["RootDB>1"]) if not pd.isna(row["RootDB>1"]) else "-"
        heur_opt = int(row["HeurOpt"]) if not pd.isna(row["HeurOpt"]) else "-"
        impr_pb = int(row["ImprPB"]) if not pd.isna(row["ImprPB"]) else "-"

        # Bold the best (full) solver
        if "bnp-full" in solver:
            analysis_latex += f"    {solver} & \\textbf{{{solved}}} & \\textbf{{{at_root}}} & {root_lp} & \\textbf{{{heur_opt}}} & {impr_pb} \\\\\n"
            analysis_latex += "    \\midrule\n"
        elif "bnc-sec" in solver:
            analysis_latex += f"    {solver} & {solved} & {at_root} & {root_lp} & {heur_opt} & {impr_pb} \\\\\n"
        else:
            analysis_latex += f"    {solver} & {solved} & {at_root} & {root_lp} & {heur_opt} & {impr_pb} \\\\\n"

    analysis_latex += "    \\bottomrule\n\\end{tabular}\n"

    analysis_path = os.path.join(paper_dir, "instance-analysis.tex")
    with open(analysis_path, "w") as f:
        f.write(analysis_latex)
    print(f"Wrote instance analysis table to {analysis_path}")

    # Table 4: Numerical safety comparison (safe vs unsafe)
    unsafe_df = summary_df[summary_df["Solver"].str.contains("unsafe|27_full$|compact", regex=True)].copy()
    if len(unsafe_df) > 0:
        unsafe_solver_map = {
            "compact": "\\textsc{bnc-sec}",
            "27_full": "\\textsc{bnp-safe}",
            "27_full_unsafe": "\\textsc{bnp-unsafe}",
            "27_full_unsafe_nobestsol": "\\textsc{bnp-unsafe-nobestsol}",
        }
        unsafe_df["Solver"] = unsafe_df["Solver"].map(lambda x: unsafe_solver_map.get(x, x))
        unsafe_df = unsafe_df[~unsafe_df["Solver"].str.contains("Virtual", case=False)]

        # Get safe time for ratio calculation
        safe_time = unsafe_df[unsafe_df["Solver"] == "\\textsc{bnp-safe}"]["Time(SGM)"].values
        safe_time = safe_time[0] if len(safe_time) > 0 else 1.0

        unsafe_df = unsafe_df[["Solver", "Solved", "Time(SGM)", "AtRoot", "HeurOpt"]]

        # Build LaTeX table
        unsafe_latex = "\\small\n\\begin{tabular}{lrrrr}\n    \\toprule\n"
        unsafe_latex += "    Solver & Solved & Time [s] & At Root & Heur Opt \\\\\n"
        unsafe_latex += "    \\midrule\n"

        for _, row in unsafe_df.iterrows():
            solver = row["Solver"]
            solved = int(row["Solved"])
            time_val = row["Time(SGM)"]
            at_root = int(row["AtRoot"]) if not pd.isna(row["AtRoot"]) else "-"
            heur_opt = int(row["HeurOpt"]) if not pd.isna(row["HeurOpt"]) else "-"

            if "bnp-safe" in solver and "unsafe" not in solver:
                unsafe_latex += f"    {solver} & {solved} & {time_val:.1f} & {at_root} & {heur_opt} \\\\\n"
            elif "bnp-unsafe" in solver and "nobestsol" not in solver:
                unsafe_latex += f"    {solver} & \\textbf{{{solved}}} & \\textbf{{{time_val:.1f}}} & {at_root} & {heur_opt} \\\\\n"
            else:
                unsafe_latex += f"    {solver} & {solved} & {time_val:.1f} & {at_root} & {heur_opt} \\\\\n"

        unsafe_latex += "    \\bottomrule\n\\end{tabular}\n"

        unsafe_path = os.path.join(paper_dir, "numerical-safety.tex")
        with open(unsafe_path, "w") as f:
            f.write(unsafe_latex)
        print(f"Wrote numerical safety table to {unsafe_path}")

    # Table 5: Cutting planes comparison
    cuts_df = summary_df[summary_df["Solver"].str.contains("27_full$|27_clique$|27_srow|compact", regex=True)].copy()
    if len(cuts_df) > 0:
        cuts_solver_map = {
            "compact": "\\textsc{bnc-sec}",
            "27_full": "No cuts",
            "27_clique": "+CC",
            "27_srow": "+SRC",
            "27_srow_clique": "+SRC+CC",
            "27_srow_pricing": "+SRC (pricing)",
        }
        cuts_df["Solver"] = cuts_df["Solver"].map(lambda x: cuts_solver_map.get(x, x))
        cuts_df = cuts_df[~cuts_df["Solver"].str.contains("Virtual", case=False)]

        # Get nocuts time for ratio calculation
        nocuts_time = cuts_df[cuts_df["Solver"] == "No cuts"]["Time(SGM)"].values
        nocuts_time = nocuts_time[0] if len(nocuts_time) > 0 else 1.0

        cuts_df = cuts_df[["Solver", "Solved", "Time(SGM)", "AtRoot"]]
        cuts_df["Ratio"] = cuts_df["Time(SGM)"] / nocuts_time

        # Build LaTeX table
        cuts_latex = "\\small\n\\begin{tabular}{lrrrr}\n    \\toprule\n"
        cuts_latex += "    Solver & Solved & Time [s] & Ratio & At Root \\\\\n"
        cuts_latex += "    \\midrule\n"

        for _, row in cuts_df.iterrows():
            solver = row["Solver"]
            solved = int(row["Solved"])
            time_val = row["Time(SGM)"]
            ratio = row["Ratio"]
            at_root = int(row["AtRoot"]) if not pd.isna(row["AtRoot"]) else "-"

            if "pricing" in solver:
                cuts_latex += f"    {solver} & \\textbf{{{solved}}} & \\textbf{{{time_val:.1f}}} & \\textbf{{{ratio:.2f}}} & {at_root} \\\\\n"
            elif "bnc-sec" in solver:
                cuts_latex += f"    {solver} & {solved} & {time_val:.1f} & {ratio:.2f} & {at_root} \\\\\n"
            else:
                cuts_latex += f"    {solver} & {solved} & {time_val:.1f} & {ratio:.2f} & {at_root} \\\\\n"

        cuts_latex += "    \\bottomrule\n\\end{tabular}\n"

        cuts_path = os.path.join(paper_dir, "cutstable.tex")
        with open(cuts_path, "w") as f:
            f.write(cuts_latex)
        print(f"Wrote cutting planes table to {cuts_path}")

    # bar_time(csv_files, optimal_counts)
    # print(avg_times, avg_nodes)
    # bar_time(csv_files, avg_times)
    # bar_time(csv_files, avg_nodes)
    # bar_time(csv_files, avg_gap)

    # TABLES GENERATION
    # instance_table()
    # solvers = [file_name_map[x] if x in file_name_map else x for x in csv_files]
    # best_time = min(avg_times)
    # ratios = list(map(lambda x: x / best_time, avg_times))

    # print(ratios)
    # aggregated_table("trieq", solvers, avg_times, ratios, optimal_counts)
    # gap_at_root_compare()

    # plot_solved("/Users/mohammedghannam/papers/lccp-paper/plot-bad.pdf")
    # time.sleep(2)
    # plot_solved("/Users/mohammedghannam/papers/lccp-paper/plot.pdf")
    # print(big_df["final_gap"].mean())
    # big_df["input_nodes"] = big_df["instance_name"].map(lambda x: extract_number_from_filename(x)) 
    # big_df.sort_values(by=["input_nodes"], inplace=True)
    # import plotly.express as px
    # fig = px.bar(big_df, x="instance_name", y="total_time", color="solver", barmode="group", log_y=False)
    # fig = px.scatter(big_df, x="input_nodes", y="total_time", color="solver", log_y=True, trendline="lowess")
    # fig = px.bar(big_df, x="instance_name", y="primal_bound", color="solver", barmode="group", log_y=False)
    # fig.show()

    # df = pd.read_csv("results20_1504simpbranching.csv")
    # print(len(df))
    # df = df[(df["final_gap"] == 0)]
    # print(len(df[(df.primal_bound - df.dual_bound_at_root ) >= 1]))

    # diff_obj_val(csv_files[-2], csv_files[-1])


    # get_best_known_solutions(["results/18_heur_rf.csv"])

    # Validate results against best known solutions
    if args.validate:
        print("\n" + "="*table_width)
        print("VALIDATION".center(table_width))
        print("="*table_width)
        for file in csv_files:
            if "compact" in file:
                continue
            validate_against_best_known(file)