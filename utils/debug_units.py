import argparse
import os

import numpy as np
from netCDF4 import Dataset

parser = argparse.ArgumentParser(description="Check variable units in NetCDF data files.")
parser.add_argument("base_path", help="Directory containing the NetCDF files")
parser.add_argument(
    "--files",
    nargs="+",
    default=["l:rico.l.nc", "q:rico.q.nc", "t:rico.t.nc", "p:rico.p.nc"],
    help="Variable:filename pairs to check (default: RICO l/q/t/p)",
)
args = parser.parse_args()

base_file_path = args.base_path
files_to_check = {}
for entry in args.files:
    var_key, file_name = entry.split(":", 1)
    files_to_check[var_key] = file_name

for var_key, file_name in files_to_check.items():
    file_path = os.path.join(base_file_path, file_name)
    print(f"\n--- Checking {file_name} ---")
    try:
        with Dataset(file_path, "r") as dataset:
            # Determine variable name
            # main.py suggests: 'l'->'l', 'q'->'q', 't'->'t', 'p'->'p'
            target_var = var_key

            if target_var not in dataset.variables:
                print(f"Variable '{target_var}' not found. Available keys: {list(dataset.variables.keys())}")
                # Try to find a likely candidate
                for key in dataset.variables:
                    if key not in ["xt", "yt", "zt", "time"]:
                        target_var = key
                        print(f"Using variable '{target_var}' instead.")
                        break

            if target_var in dataset.variables:
                data_var = dataset.variables[target_var]
                print(f"Variable: {target_var}")
                print(f"Units: {getattr(data_var, 'units', 'No units attribute')}")

                # Read first timestep
                data = data_var[0, :, :, :]
                print(f"Shape: {data.shape}")
                print(f"Min: {np.min(data)}")
                print(f"Max: {np.max(data)}")
                print(f"Mean: {np.mean(data)}")

                # Heuristic check for units
                if var_key in ["l", "q"]:
                    if np.max(data) > 1.0:
                        print("-> Likely g/kg (values > 1)")
                    else:
                        print("-> Likely kg/kg (values < 1)")
            else:
                print(f"Could not identify main variable in {file_name}.")

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred processing {file_name}: {e}")
