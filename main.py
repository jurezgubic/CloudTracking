import argparse, os, gc, time
from memory_profiler import profile
from src.data_management import load_cloud_field_from_file, calculate_mean_velocities
from src.netcdf_writer import write_cloud_tracks_to_netcdf
from lib.cloudtracker import CloudTracker
import src.data_management as data_management
import numpy as np
from netCDF4 import Dataset  # Needed for finalizing

# Warning: user needs to modify: base_file_path, file_name, output_netcdf_path, total_timesteps and config

# Set file paths and parameters
base_file_path = '/Users/jure/PhD/coding/RICO_1hr/'

# paths to the LES data files
file_name = {
    'l': 'rico.l.nc',
    'u': 'rico.u.nc',
    'v': 'rico.v.nc',
    'w': 'rico.w.nc',
    'p': 'rico.p.nc',
    't': 'rico.t.nc',
    'q': 'rico.q.nc',
}

# Set output file path
output_netcdf_path = 'cloud_results.nc'

# Set number of timesteps to process
total_timesteps = 3

# Set configuration parameters
config = {
    'min_size': 100,  # Minimum size of cloud objects to be considered
    'l_condition': 0.0007, # kg/kg. Minimum threshold for liquid water
    'w_condition': 0.0,  # m/s. Minimum condition for vertical velocity
    'w_switch': True,  # True if you want to use vertical velocity threshold
    'timestep_duration': 60,  # Duration between timesteps in seconds
    'distance_threshold': 0, # Max dist between merging clouds across boundary
    'plot_switch': False, # Plot cloud field at each timestep
    'v_drift': -4.0, # -4. m/s, taken from RICO namelist
    'u_drift': -5.0, # -5. m/s, taken from RICO namelist
    'horizontal_resolution': 25.0, # m, taken from namelist
    'switch_background_drift': False, # True if you want to subtract the background drift
    'switch_wind_drift': True, # True if you want to subtract the wind drift
    'switch_vertical_drift': True,  # Enable vertical drift consideration
    'cloud_base_altitude': 700, # m, from input data analysis
}

# Add this function to calculate mean_w
def calculate_mean_vertical_velocity(file_path, file_names, timestep):
    """Calculate mean vertical velocity for each z-level at the given timestep."""
    w_data = data_management.load_w_field(file_path, file_names['w'], timestep)
    mean_w = np.mean(w_data, axis=(1, 2))  # Average over y and x dimensions
    return mean_w

# @profile
def process_clouds(cloud_tracker):
    for timestep in range(total_timesteps):
        start = time.time() # Start timer
        print ("-"*50)
        print(f"Processing timestep {timestep+1} of {total_timesteps}")

        # Load cloud field for the current timestep
        cloud_field = data_management.load_cloud_field_from_file(base_file_path, file_name, timestep, config)

        # Calculate mean velocities for the current timestep
        mean_u, mean_v = calculate_mean_velocities(base_file_path, file_name, timestep)
        mean_w = calculate_mean_vertical_velocity(base_file_path, file_name, timestep)  # Calculate mean vertical velocity

        # Track clouds across timesteps
        cloud_tracker.update_tracks(cloud_field, mean_u, mean_v, mean_w, cloud_field.zt)  # Pass mean_w

        # Write cloud track information to netCDF
        zt = cloud_field.zt
        write_cloud_tracks_to_netcdf(cloud_tracker.get_tracks(), output_netcdf_path, timestep, zt)

        # Force garbage collection to avoid memory issues
        gc.collect()
        stop = time.time() # Stop timer
        print (f"Time taken: {(stop-start)/60:.1f} minutes")
    print("Cloud tracking complete.")

    # After all timesteps, flag partial-lifetime clouds
    finalize_partial_lifetime_tracks(cloud_tracker, total_timesteps)


def finalize_partial_lifetime_tracks(cloud_tracker, total_timesteps):
    """Flag partial-lifetime tracks in the NetCDF so they can be ignored in analyses."""
    partial_ids = []
    for t_id, track in cloud_tracker.get_tracks().items():
        if not track:
            continue
        t_first = track[0].timestep
        t_last = track[-1].timestep
        # Check if it started at the first step or is active at the last step
        if t_first == 0 or (track[-1].is_active and t_last == total_timesteps - 1):
            partial_ids.append(t_id)

    if not partial_ids:
        print("No partial-lifetime tracks found.")
        return

    print(f"Flagging {len(partial_ids)} partial-lifetime track(s) as invalid...")
    with Dataset(output_netcdf_path, "r+") as ds:
        valid_var = ds.variables['valid_track']
        for i, t_id in enumerate(cloud_tracker.get_tracks().keys()):
            # If t_id happens to be in partial_ids, set valid_track to 0
            if t_id in partial_ids:
                valid_var[i] = 0


def main(delete_existing_file):
    # Delete the existing output file if the option is provided
    if delete_existing_file and os.path.exists(output_netcdf_path):
        os.remove(output_netcdf_path)
        print(f"Deleted existing file: {output_netcdf_path}")

    # Initialize CloudTracker
    cloud_tracker = CloudTracker(config)

    # Process clouds
    process_clouds(cloud_tracker)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cloud tracking script.")
    parser.add_argument('--delete', action='store_true', help="Delete the existing output netcdf file before running the script.")
    args = parser.parse_args()

    main(args.delete)


## set proper output file path (placeholder for later)
# import os
# import datetime
# today = datetime.date.today()
# output_folder = f'output/{today}'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# output_netcdf_path = f'{output_folder}/{total_timesteps}timesteps_{config["l_condition"]}l_condition.nc'
# print(f"Output netcdf file path: {output_netcdf_path}")


