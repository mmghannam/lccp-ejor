import re
import sys
import json 

file_path = sys.argv[1]

# Initialize variables to store the extracted numerical values
num_nodes = None
time_seconds = None
primal_bound = None
dual_bound = None
gap_percentage = None
dual_bound_at_root = None
gap_at_root = None
primal_bound_at_root = None
# Regular expressions for extracting numerical values
node_time_regex = r"Explored (\d+) nodes .* in (\d+\.\d+) seconds"
objective_bound_regex = r"Best objective ([\de\+\-\.]+), best bound ([\de\+\-\.]+), gap ([\de\+\-\.]+)%"

# Read the file to extract the required numerical values
started_log = False

with open(file_path, 'r') as file:

    for line in file:
        if "  " in line:
            try:
                split_line = [x.strip() for x in line.split(" ") if x.strip()]
                # print(split_line)
                nodes,open_nodes,*_,pb,db,gap, _, time = split_line
                if num_nodes is None:
                    num_nodes = int(nodes)
                if primal_bound is None:
                    primal_bound = float(pb)
                if dual_bound is None:
                    dual_bound = float(db)
                if gap_percentage is None:
                    gap_percentage = float(gap[:-1])
                if nodes == '0' and open_nodes != '0':
                    # print("db at root", dual_bound)
                    dual_bound_at_root = float(db)
            except Exception as e:
                print(e)

        if "Explored" in line:
            match = re.search(node_time_regex, line)
            if match:
                num_nodes = int(match.group(1))
                time_seconds = float(match.group(2))
        if "Best objective" in line:
            match = re.search(objective_bound_regex, line)
            if match:
                primal_bound = float(match.group(1))
                dual_bound = float(match.group(2))
                gap_percentage = float(match.group(3))
        if "Loaded user MIP start with objective" in line:
            primal_bound_at_root = float(line.split(" ")[-1].strip())
        if "Root relaxation:" in line:
            dual_bound_at_root = line.split(",")[0].split(" ")[-1].strip()
            if dual_bound_at_root == "cutoff":
                dual_bound_at_root = primal_bound_at_root
            else:
                try:
                    dual_bound_at_root = float(dual_bound_at_root)
                except ValueError:
                    # e.g. "infeasible" root relaxation (instance solved in presolve);
                    # leave the at-root dual bound unknown rather than crashing.
                    dual_bound_at_root = None

# Display the extracted numerical values
num_nodes, time_seconds, primal_bound, dual_bound, gap_percentage


# Save data to a dictionary
data_dict = {
    # "time_to_first_node": time_to_solve_root,
    "dual_bound_at_root": dual_bound_at_root,
    "gap_at_root": gap_at_root,
    "primal_bound_at_root": primal_bound_at_root,
    "final_gap": gap_percentage,
    "primal_bound": primal_bound,
    "dual_bound": dual_bound,
    "number_of_nodes": num_nodes,
    "instance_name": file_path.split("/")[-1].split(".")[0],
    "timelimit": 7200,
    "total_time": time_seconds,
    # "solution": [line.split("1.0")[0].strip() for line in solution_lines]
}

print(data_dict)

# Save the dictionary to a JSON file
output_file = f"{file_path.split('.')[0]}.json"
with open(output_file, "w") as f:
    json.dump(data_dict, f, indent=4)
