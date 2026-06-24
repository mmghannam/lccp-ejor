import json
import os
import csv
import glob
import sys

# Specify the directory and output CSV file
directory = sys.argv[1]
if directory[-1] == "/":
    directory = directory[:-1]
    
output_csv = f"{directory}.csv"

# List of fields to extract from the JSON files
fields = [
    "instance_name",
    "time_to_first_node",
    "dual_bound_at_root",
    "primal_bound_at_root",
    "first_primal_bound",
    "final_gap",
    "primal_bound",
    "dual_bound",
    "number_of_nodes",
    "timelimit",
    "total_time",
    "cg_iterations_total",
    "cg_iterations_root",
]

# Find all JSON files in the directory
json_files = glob.glob(os.path.join(directory, "*.json"))

# Collect data from JSON files
data = []
for json_file in json_files:
    with open(json_file, "r") as f:
        json_data = json.load(f)
        data.append({field: json_data.get(field, None) for field in fields})

# Write data to the CSV file
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(data)
