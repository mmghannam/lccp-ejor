#!/bin/bash

# Specify the Python script
PYTHON_SCRIPT="analysis/extract_info.py"

# Iterate through all input arguments
for dir in "$@"
do
  # Check if directory exists (skip files like .csv from previous runs)
  if [ -d "${dir}" ]; then
    # Iterate through all files in the directory
    for file in "${dir}"/*
    do
      # Call the Python script with the current file as an argument
      python "${PYTHON_SCRIPT}" "${file}"
    done
    python analysis/aggregator.py "${dir}"
  else
    echo "Skipping ${dir} (not a directory)"
  fi
done

