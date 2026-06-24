import re
import sys
import json

paths = sys.argv[1:]
# path = "results/att48.dat.sol"

for path in paths:
    # Open log file
    with open(path, "r") as f:
        content = f.read()

    # Extract time to first node
    time_matches = re.findall(r"(\d+\.\d+)s\| +1 \|", content)
    time_value = float(time_matches[-1]) if time_matches else None

    # Extract final gap
    gap_match = re.search(r"Gap\s+:\s+(\d+(\.\d+)?) %", content)
    gap_value = float(gap_match.group(1)) if gap_match else None

    # Extract primal bound
    primal_match = re.search(r"Primal Bound\s+:\s+([-+]?\d*\.\d+([eE][-+]?\d+)?)", content)
    primal_value = float(primal_match.group(1)) if primal_match else None

    # Extract dual bound
    dual_match = re.search(r"Dual Bound\s+:\s+([-+]?\d*\.\d+([eE][-+]?\d+)?)", content)
    dual_value = float(dual_match.group(1)) if dual_match else None

    # Extract number of nodes
    nodes_match = re.search(r"Solving Nodes\s+:\s+(\d+)", content)
    nodes_value = int(nodes_match.group(1)) if nodes_match else None

    total_time= None
    time_to_solve_root = None
    dual_value_at_root = None
    primal_bound_at_root = None
    first_primal_bound = None

    # Count CG iterations (total and at root)
    cg_iterations_total = 0
    cg_iterations_root = 0
    seen_first_iter1 = False
    root_done = False
    for line in content.splitlines():
        iter_match = re.match(r'-- iter#(\d+)', line)
        if iter_match:
            iter_num = int(iter_match.group(1))
            cg_iterations_total += 1
            if not seen_first_iter1:
                seen_first_iter1 = True
                cg_iterations_root = 1
            elif iter_num == 1:
                # Reset to iter#1 means we moved to a new node
                root_done = True
            elif not root_done:
                cg_iterations_root += 1

    for line in content.splitlines():
        if "|" in line:
            data = line.split("|")
            if len(data) > 4:
                time, node, *_, db,pb, gap,_ = data
                if "time" in time: continue
                
                try:
                    x = float(pb)
                    x = float(db)
                except ValueError:
                    continue

                if first_primal_bound is None:
                    first_primal_bound = float(pb)

                if node.strip() == "1":
                    try:
                        dual_value_at_root = float(db)
                        time_to_solve_root = time = re.findall(r'(\d+\.\d+s).*', time)[0]
                        primal_bound_at_root = float(pb)
                    except:
                        pass
                
                
                if primal_value is None or float(pb) < primal_value:
                    primal_value = float(pb)
                if dual_value is None or dual_value_at_root is not None or float(db) > dual_value:
                    dual_value = float(db) if "--" not in db else dual_value_at_root
                if nodes_value is None:
                    nodes_value = int(node)
                if gap_value is None:
                    gap_value = gap.strip()[:-1]
        elif "Solving Time" in line:
            total_time = float(line.split(":")[-1].strip())

    # TODO: Extract number of variables

    # Extract instance path
    instance_path = path.split("/")[-1]

    # Extract timelimit
    timelimit_match = re.search(r"Timelimit:\s+(\d+)", content)
    timelimit = int(timelimit_match.group(1)) if timelimit_match else None

    print(f"Time to first node: {time_to_solve_root}")
    print(f"Dualboud at root: {dual_value_at_root}")
    print(f"Final gap: {gap_value}%")
    print(f"Primal bound: {primal_value}")
    print(f"Dual bound: {dual_value}")
    print(f"Number of nodes: {nodes_value}")
    print(f"Instance path: {instance_path}")
    print(f"Timelimit: {timelimit}")
    print(f"Total time: {total_time}")
    # print(f"Number of variables: {vars_value}")

    # Extract final solution
    solution_lines = []
    solution_start = False
    for line in content.splitlines():
        if line.startswith("Best solution found:"):
            solution_start = True
        elif solution_start:
            if line.strip() == "":
                break
            else:
                solution_lines.append(line.strip())


    # Save data to a dictionary
    data_dict = {
        "time_to_first_node": time_to_solve_root,
        "dual_bound_at_root": dual_value_at_root,
        "primal_bound_at_root": primal_bound_at_root,
        "first_primal_bound": first_primal_bound,
        "final_gap": gap_value,
        "primal_bound": primal_value,
        "dual_bound": dual_value,
        "number_of_nodes": nodes_value,
        "instance_name": instance_path.split("/")[-1].split(".")[0],
        "timelimit": timelimit,
        "total_time": total_time,
        "cg_iterations_total": cg_iterations_total,
        "cg_iterations_root": cg_iterations_root,
        "solution": [line.split("1.0")[0].strip() for line in solution_lines]
    }

    # Save the dictionary to a JSON file
    output_file = f"{path.split('.')[0]}.json"
    with open(output_file, "w") as f:
        json.dump(data_dict, f, indent=4)