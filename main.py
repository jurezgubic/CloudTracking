from memory_profiler import profile
import gc
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
    'w': 'rico.w.nc'
}

# Set output file path
output_netcdf_path = 'cloud_results.nc'

# Set number of timesteps to process
total_timesteps = 3

# Set configuration parameters
config = {
    'min_size': 50,  # Minimum size of cloud objects to be considered
    'l_condition': 0.0008, # Minimum threshold for liquid water
    'w_condition': 1.0,  # Minimum condition for vertical velocity
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
        print ("-"*50)
        print(f"Processing timestep {timestep+1} of {total_timesteps}")

        # Load cloud field for the current timestep
        cloud_field = data_management.load_cloud_field_from_file(base_file_path, file_name, timestep, config)

        # Calculate mean velocities for the current timestep
        mean_u, mean_v = calculate_mean_velocities(base_file_path, file_name, timestep)

        # Track clouds across timesteps
        cloud_tracker.update_tracks(cloud_field, mean_u, mean_v, cloud_field.zt)

        # Write cloud track information to netCDF
        write_cloud_tracks_to_netcdf(cloud_tracker.get_tracks(), output_netcdf_path, timestep)

         # Force garbage collection to avoid memory issues
        gc.collect()
    print("Cloud tracking complete.")


def main():
    # Initialize CloudTracker
    cloud_tracker = CloudTracker(config)

    # Process clouds
    process_clouds(cloud_tracker)

if __name__ == "__main__":
    main()


## set proper output file path (placeholder for later)
# import os
# import datetime
# today = datetime.date.today()
# output_folder = f'output/{today}'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# output_netcdf_path = f'{output_folder}/{total_timesteps}timesteps_{config["l_condition"]}l_condition.nc'
# print(f"Output netcdf file path: {output_netcdf_path}")


