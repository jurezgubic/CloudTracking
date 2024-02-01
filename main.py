from src.data_management import load_cloud_field_from_file
from lib.cloudtracker import CloudTracker
from src.netcdf_writer import write_cloud_data_to_netcdf

# Set file paths and parameters
l_file_path = '/Users/jure/PhD/coding/RICO_1hr/rico.l.nc'
w_file_path = '/Users/jure/PhD/coding/RICO_1hr/rico.w.nc'

# set parameters
total_timesteps = 4 # Total number of timesteps to process


config = {
    'min_size': 50,  # Minimum size of cloud objects to be considered
    'l_condition': 0.001,  # Threshold condition for liquid water
    'consider_w': False  # Whether to consider w
}

# Initialize CloudTracker
cloud_tracker = CloudTracker()

# Process each timestep
for timestep in range(total_timesteps):
    print(f"Processing timestep {timestep}")

    cloud_field = load_cloud_field_from_file(l_file_path, w_file_path, timestep, config)

    # Update cloud tracks with the current cloud field
    cloud_tracker.update_tracks(cloud_field)

    # remove the old cloud field from memory
    del cloud_field

print("Cloud tracking complete.")

# Write cloud data to NetCDF
netcdf_filename = 'cloud_data.nc'
write_cloud_data_to_netcdf(cloud_tracker.get_tracks(), netcdf_filename)

print("NetCDF file written.")

