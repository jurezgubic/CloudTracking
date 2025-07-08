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
total_timesteps = 7

# Set configuration parameters
config = {
    'min_size': 10,  # Minimum size of cloud objects to be considered
    'l_condition': 0.0009, # kg/kg. Minimum threshold for liquid water
    'w_condition': 0.0,  # m/s. Minimum condition for vertical velocity
    'w_switch': False,  # True if you want to use vertical velocity threshold
    'timestep_duration': 60,  # Duration between timesteps in seconds
    'distance_threshold': 0, # Max dist between merging clouds across boundary
    'plot_switch': False, # Plot cloud field at each timestep
    'horizontal_resolution': 25.0, # m, taken from namelist
    'switch_wind_drift': True, # True if you want to subtract the wind drift
    'switch_vertical_drift': True,  # Enable vertical drift consideration
    'cloud_base_altitude': 700, # m, from input data analysis
}

# Function to calculate mean_w
def calculate_mean_vertical_velocity(file_path, file_names, timestep):
    """Calculate mean vertical velocity for each z-level at the given timestep."""
    w_data = data_management.load_w_field(file_path, file_names['w'], timestep)
    mean_w = np.mean(w_data, axis=(1, 2))  # Average over y and x dimensions
    return mean_w

# @profile
def process_clouds(cloud_tracker):
    partial_lifetime_cloud_counter = 0  # Initialize counter
    
    # Track all cloud IDs that are "tainted" by having ancestry from timestep 1
    tainted_lineage = set()
    
    for timestep in range(total_timesteps):
        start = time.time()  # Start timer
        print("-"*50)
        print(f"Processing timestep {timestep+1} of {total_timesteps}")

        # Load cloud field for the current timestep
        cloud_field = data_management.load_cloud_field_from_file(base_file_path, file_name, timestep, config)

        # Calculate mean velocities for the current timestep
        mean_u, mean_v = calculate_mean_velocities(base_file_path, file_name, timestep)
        mean_w = calculate_mean_vertical_velocity(base_file_path, file_name, timestep)

        # Track clouds across timesteps
        cloud_tracker.update_tracks(cloud_field, mean_u, mean_v, mean_w, 
                                   cloud_field.zt, cloud_field.xt, cloud_field.yt)

        # Special handling for each timestep:
        if timestep == 0:
            # First timestep: All clouds are "tainted" because we didn't observe their birth
            timestep_0_ids = set(cloud_tracker.cloud_tracks.keys())
            tainted_lineage.update(timestep_0_ids)
            
            # Remove these partial tracks from the tracker before writing
            partial_tracks = list(timestep_0_ids)
            partial_lifetime_cloud_counter += len(partial_tracks)
            
            for track_id in partial_tracks:
                del cloud_tracker.cloud_tracks[track_id]
                
            print(f"Removed {len(partial_tracks)} partial lifetime clouds from first timestep")
            
        else:
            # Identify clouds that are "tainted" by descent from timestep 1 clouds
            newly_tainted = []
            
            for track_id, track in cloud_tracker.cloud_tracks.items():
                # Skip already tainted
                if track_id in tainted_lineage:
                    continue
                
                # Check if this cloud has any parent in the tainted set
                for cloud in track:
                    if hasattr(cloud, 'merged_into') and cloud.merged_into in tainted_lineage:
                        newly_tainted.append(track_id)
                        break
            
            # Update the tainted set
            tainted_lineage.update(newly_tainted)
            
            # Remove newly tainted tracks
            if newly_tainted:
                partial_lifetime_cloud_counter += len(newly_tainted)
                print(f"Removing {len(newly_tainted)} clouds continuing from timestep 1")
                
                for track_id in newly_tainted:
                    del cloud_tracker.cloud_tracks[track_id]

        # Write cloud track information to netCDF (now only writing complete lifecycle clouds)
        zt = cloud_field.zt
        write_cloud_tracks_to_netcdf(cloud_tracker.get_tracks(), output_netcdf_path, timestep, zt)

        # Force garbage collection to avoid memory issues
        gc.collect()
        stop = time.time()  # Stop timer
        print(f"Time taken: {(stop-start)/60:.1f} minutes")
    
    # After all timesteps, identify and remove clouds still active at the end
    final_partial_tracks = [tid for tid, track in cloud_tracker.cloud_tracks.items() 
                           if track[-1].is_active and track[-1].timestep == total_timesteps - 1]
    partial_lifetime_cloud_counter += len(final_partial_tracks)
    
    # Remove these final partial tracks
    for track_id in final_partial_tracks:
        del cloud_tracker.cloud_tracks[track_id]
    
    print("-"*50)
    print(f"Total partial lifetime clouds removed: {partial_lifetime_cloud_counter}")
    print("Cloud tracking complete.")

    # Write the final state with only complete lifecycle clouds
    write_cloud_tracks_to_netcdf(cloud_tracker.get_tracks(), output_netcdf_path, total_timesteps-1, zt)


def finalize_partial_lifetime_tracks(cloud_tracker, total_timesteps):
    """Flag partial-lifetime tracks in the NetCDF so they can be ignored in analyses."""
    partial_ids = []
    merged_ids = []
    
    for t_id, track in cloud_tracker.get_tracks().items():
        if not track:
            continue
        
        t_first = track[0].timestep
        t_last = track[-1].timestep
        last_cloud = track[-1]
        
        # Check for partial lifetime
        if t_first == 0 or (last_cloud.is_active and t_last == total_timesteps - 1):
            partial_ids.append(t_id)
        
        # Check for merged clouds
        if not last_cloud.is_active and last_cloud.merged_into is not None:
            merged_ids.append(t_id)

    # Update flags in the NetCDF
    with Dataset(output_netcdf_path, "r+") as ds:
        valid_var = ds.variables['valid_track']
        merged_var = ds.variables['merged_into']
        
        for i, t_id in enumerate(cloud_tracker.get_tracks().keys()):
            # Handle partial lifetime flags
            if t_id in partial_ids:
                valid_var[i] = 0
            
            # Note: merged_into is already handled in the write_cloud_tracks_to_netcdf function


def main(delete_existing_file):
    # Delete the existing output file if the option is provided
    if delete_existing_file and os.path.exists(output_netcdf_path):
        os.remove(output_netcdf_path)
        print(f"Deleted existing file: {output_netcdf_path}")

    # Initialize CloudTracker
    cloud_tracker = CloudTracker(config)

    # Process clouds - partial lifetime clouds will be removed inside this function
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


