#!/bin/bash

# Specify the Python script
PYTHON_SCRIPT="extract_info_compact.py"

# Iterate through all input arguments
for dir in "$@"
do
  # Check if directory exists
  if [ -d "${dir}" ]; then
    # Iterate through all files in the directory
    for file in "${dir}"/*.npcc.sol
    do
      # Call the Python script with the current file as an argument
      python "${PYTHON_SCRIPT}" "${file}"
    done
  else
    echo "Directory ${dir} does not exist."
  fi
  python aggregator.py "${dir}"
done

