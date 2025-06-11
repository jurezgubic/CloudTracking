import os
import glob
import argparse # Ensure argparse is imported
import numpy as np # Ensure numpy is imported
from netCDF4 import Dataset # <--- ADD THIS LINE
from src.data_management import load_cloud_field_from_file, calculate_mean_velocities, load_w_field
from src.netcdf_writer import write_cloud_tracks_to_netcdf
from lib.cloudtracker import CloudTracker
import src.data_management as data_management
import gc
import time

# Warning: user needs to modify: base_file_path, output_netcdf_path, total_timesteps and config

# Set file paths and parameters
base_file_path = '/Users/jure/PhD/coding/LBA_sample_data/jun10' # MODIFIED: Update to LBA data path

# Set output file path
output_netcdf_path = 'lba_cloud_results.nc' # MODIFIED: Example new output name

# Set number of timesteps to process (optional cap, if 0 or high, processes all found files)
total_timesteps = 0 # MODIFIED: Example: 0 to process all found files, or set a specific number

# Set configuration parameters
config = {
    'min_size': 10,
    'l_condition': 0.001, # kg/kg. Minimum threshold for liquid water (adjust as LBA is kg/kg)
    'w_condition': 0.0,
    'w_switch': False,
    'timestep_duration': 180,  # Duration between LBA timesteps in seconds
    'distance_threshold': 0,
    'plot_switch': False,
    # 'v_drift': -4.0, # Remove or set to 0 if not applicable/known for LBA
    # 'u_drift': -5.0, # Remove or set to 0 if not applicable/known for LBA
    'horizontal_resolution': 200.0, # m, taken from namelist - VERIFY FOR LBA
    'switch_background_drift': False,
    'switch_wind_drift': True,
    'switch_vertical_drift': True,
    'cloud_base_altitude': 700, # m, from input data analysis - VERIFY FOR LBA

    # LBA specific config
    'dataset_type': 'LBA', # Added to help functions distinguish
    'lba_file_pattern': '3dfields_ts_*.nc', # Pattern to find LBA files
    'lba_var_map': {
        'l': 'q_cloud_liquid_mass',
        'w': 'w',
        'p': 'p',
        't': 'th', # Assuming 'th' is potential temperature
        'q_ice': 'q_ice_mass', # For calculating total water (liq+ice)
        'u': 'u',
        'v': 'v',
        # Add 'q_vapour': 'q_vapour' if needed by physics beyond q_t
    },
    'lba_coord_names': { # Names of coordinate variables in LBA files
        'x': 'x',
        'y': 'y',
        'z': 'z' # Using 'z' as the primary vertical coordinate as requested
    }
}

# Function to calculate mean_w
def calculate_mean_vertical_velocity(lba_file_path, config_params): # MODIFIED signature
    """Calculate mean vertical velocity for each z-level from the given LBA timestep file."""
    # Use the updated load_w_field from data_management
    w_data = load_w_field(lba_file_path, config_params) # MODIFIED call
    mean_w = np.mean(w_data, axis=(1, 2))  # Average over y and x dimensions
    return mean_w

# @profile
def process_clouds(cloud_tracker, lba_files_to_process, config_params): # MODIFIED signature
    partial_lifetime_cloud_counter = 0
    tainted_lineage = set()
    
    processed_timesteps = 0
    first_zt = None # To store zt from the first processed file

    for timestep_idx, current_lba_file in enumerate(lba_files_to_process):
        print(f"Processing LBA file: {current_lba_file} (Timestep index: {timestep_idx})")
        
        # Load cloud field for the current LBA file
        # The 'timestep_idx' is a conceptual index for tracking, not for data slicing within the file
        cloud_field = load_cloud_field_from_file(current_lba_file, timestep_idx, config_params)
        
        if cloud_field is None: # Handle case where loading might fail or return None
            print(f"Warning: Could not load cloud field from {current_lba_file}. Skipping.")
            continue

        if first_zt is None:
            first_zt = cloud_field.zt # Store zt from the first file for NetCDF initialization

        # Calculate mean velocities for the current LBA file
        mean_u, mean_v = calculate_mean_velocities(current_lba_file, config_params)
        mean_w_profile = calculate_mean_vertical_velocity(current_lba_file, config_params) # Renamed for clarity

        # Update tracks
        cloud_tracker.update_tracks(cloud_field, mean_u, mean_v, mean_w_profile,
                                    cloud_field.zt, cloud_field.xt, cloud_field.yt)
        processed_timesteps +=1
    
    # After all timesteps, identify and remove clouds still active at the end
    final_partial_tracks = [tid for tid, track in cloud_tracker.cloud_tracks.items() 
                           if track[-1].is_active and track[-1].timestep == processed_timesteps - 1] # MODIFIED to use processed_timesteps
    partial_lifetime_cloud_counter += len(final_partial_tracks)
    
    # temporarily commented out the removal of final partial tracks
    # ---------------------------------------------------
    # # Remove these final partial tracks
    # for track_id in final_partial_tracks:
    #     # Assuming you have a method to remove/mark tracks, or adjust logic here
    #     if track_id in cloud_tracker.cloud_tracks: # Check if track still exists
    #          del cloud_tracker.cloud_tracks[track_id] # Example: removing the track
    # print("-"*50)
    # print(f"Total partial lifetime clouds removed: {partial_lifetime_cloud_counter}")
    # print("Cloud tracking complete.")
    # ---------------------------------------------------

    
    if first_zt is None and processed_timesteps > 0:
        print("Error: zt was not captured, cannot write NetCDF.") # Should not happen if processing occurs
        return
    if processed_timesteps == 0:
        print("No timesteps processed. Skipping NetCDF output.")
        return

    # Write the final state with only complete lifecycle clouds
    # Ensure 'first_zt' is available and 'processed_timesteps' reflects actual processed count
    write_cloud_tracks_to_netcdf(cloud_tracker.get_tracks(), output_netcdf_path, processed_timesteps -1, first_zt)


def finalize_partial_lifetime_tracks(cloud_tracker, total_processed_timesteps): # MODIFIED param name
    """Flag partial-lifetime tracks in the NetCDF so they can be ignored in analyses."""
    partial_ids = []
    merged_ids = []
    
    for t_id, track in cloud_tracker.get_tracks().items():
        if not track: continue # Skip empty tracks
        # Check if track starts at timestep 0
        starts_at_zero = track[0].timestep == 0
        # Check if track ends at the last processed timestep and is still active
        # Note: process_clouds already removes tracks active at the very end.
        # This logic might need adjustment based on how process_clouds handles final tracks.
        # ends_at_last = track[-1].timestep == total_processed_timesteps - 1 and track[-1].is_active

        # For LBA, a track is partial if it starts after timestep 0.
        # Tracks active at the very end are already removed by process_clouds.
        if not starts_at_zero:
            partial_ids.append(t_id)
        
        # Check for merges
        for cloud_obj in track:
            if cloud_obj.merged_into is not None:
                merged_ids.append(t_id)
                break # Only need to add the track_id once

    # Update flags in the NetCDF
    if not os.path.exists(output_netcdf_path):
        print(f"Warning: Output file {output_netcdf_path} not found for finalization.")
        return
        
    with Dataset(output_netcdf_path, "r+") as ds:
        if 'valid_track' in ds.variables:
            valid_track_var = ds.variables['valid_track']
            for i, t_id_stored in enumerate(ds.variables['track_id'][:]): # Assuming you have a track_id variable
                if t_id_stored in partial_ids:
                    valid_track_var[i] = 0  # Mark as invalid/partial
        else:
            print("Warning: 'valid_track' variable not found in NetCDF.")
        
        # If you also want to flag merged tracks differently, you can add another variable or logic here.


def main(delete_existing_file):
    if delete_existing_file and os.path.exists(output_netcdf_path):
        os.remove(output_netcdf_path)
        print(f"Deleted existing output file: {output_netcdf_path}")

    # Get all LBA files, sort them numerically based on the timestamp in the filename
    # This assumes filenames like '3dfields_ts_10800.nc', '3dfields_ts_10980.nc'
    glob_pattern = os.path.join(base_file_path, config['lba_file_pattern'])
    lba_files = sorted(
        glob.glob(glob_pattern),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_ts_')[-1])
    )

    if not lba_files:
        print(f"No LBA files found in {base_file_path} matching pattern {config['lba_file_pattern']}")
        return

    # Determine timesteps to process
    actual_total_timesteps = len(lba_files)
    timesteps_to_process_count = min(total_timesteps, actual_total_timesteps) if total_timesteps > 0 else actual_total_timesteps
    
    lba_files_to_process = lba_files[:timesteps_to_process_count]

    if not lba_files_to_process:
        print("No LBA files selected for processing.")
        return
        
    print(f"Found {actual_total_timesteps} LBA files. Processing {len(lba_files_to_process)} files.")

    cloud_tracker = CloudTracker(config)
    process_clouds(cloud_tracker, lba_files_to_process, config) # MODIFIED call

    # Finalize tracks if any were processed
    if len(lba_files_to_process) > 0:
        finalize_partial_lifetime_tracks(cloud_tracker, len(lba_files_to_process))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud tracking script for LBA data.")
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


