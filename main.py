import argparse
import os
from memory_profiler import profile
import gc, time
from src.data_management import load_cloud_field_from_file, calculate_mean_velocities
from src.netcdf_writer import write_cloud_tracks_to_netcdf
from lib.cloudtracker import CloudTracker
import src.data_management as data_management

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
    'min_size': 10,  # Minimum size of cloud objects to be considered
    'l_condition': 0.001, # kg/kg. Minimum threshold for liquid water
    'w_condition': 0.0,  # m/s. Minimum condition for vertical velocity
    'w_switch': True,  # True if you want to use vertical velocity threshold
    'timestep_duration': 60,  # Duration between timesteps in seconds
    'distance_threshold': 3, # Max dist between merging clouds across boundary
    'plot_switch': False, # Plot cloud field at each timestep
    'v_drift': -4.0, # -4. m/s, taken from RICO namelist
    'u_drift': -5.0, # -5. m/s, taken from RICO namelist
    'horizontal_resolution': 25.0, # m, taken from namelist
    'switch_background_drift': False, # True if you want to subtract the background drift
    'switch_wind_drift': True, # True if you want to subtract the wind drift
    'cloud_base_altitude': 700, # m, from input data analysis
}


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

        # Track clouds across timesteps
        cloud_tracker.update_tracks(cloud_field, mean_u, mean_v, cloud_field.zt)

        # Write cloud track information to netCDF
        zt = cloud_field.zt
        write_cloud_tracks_to_netcdf(cloud_tracker.get_tracks(), output_netcdf_path, timestep, zt)

         # Force garbage collection to avoid memory issues
        gc.collect()
        stop = time.time() # Stop timer
        print (f"Time taken: {(stop-start)/60:.1f} minutes")
    print("Cloud tracking complete.")


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


