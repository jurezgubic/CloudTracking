from src.data_management import load_cloud_field_from_file
from src.netcdf_writer import write_cloud_tracks_to_netcdf
from lib.cloudtracker import CloudTracker

# Set file paths and parameters
l_file_path = '/Users/jure/PhD/coding/RICO_1hr/rico.l.nc'
total_timesteps = 3

output_netcdf_path = 'cloud_results.nc'

config = {
    'min_size': 50,  # Minimum size of cloud objects to be considered
    'l_condition': 0.0004,#0.0002  # Threshold condition for liquid water
    'timestep_duration': 60,  # Duration between timesteps in seconds
    'distance_threshold': 3, # Maximum distance between merging clouds
    'plot_switch': True  # Plot cloud field at each timestep
}


# Initialize CloudTracker
cloud_tracker = CloudTracker()

# Process each timestep
for timestep in range(total_timesteps):
    print ("-"*50)
    print(f"Processing timestep {timestep}")

    # Load cloud field for the current timestep
    cloud_field = load_cloud_field_from_file(l_file_path, timestep, config)

    # Track clouds across timesteps
    cloud_tracker.update_tracks(cloud_field)

print("Cloud tracking complete.")

# Write cloud track information to netCDF
write_cloud_tracks_to_netcdf(cloud_tracker.get_tracks(), output_netcdf_path)
print("Cloud track information written to netCDF.")

