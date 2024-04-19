from netCDF4 import Dataset
import numpy as np
import os

def initialize_netcdf(file_path):
    # Create the file and define dimensions and variables
    with Dataset(file_path, 'w', format='NETCDF4') as root_grp:
        root_grp.createDimension('track', 10000)  # Fixed large number of tracks
        root_grp.createDimension('time', None)  # Unlimited time dimension
        root_grp.createDimension('point', 100000)  # Static dimension for cloud points
        root_grp.createDimension('coordinate', 3)  # Static dimension for 3D coordinates

        root_grp.createVariable('size', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('location_x', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('location_y', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('location_z', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('cloud_points', 'f4', ('track', 'time', 'point', 'coordinate'), fill_value=np.nan)

def write_cloud_tracks_to_netcdf(tracks, file_path, timestep):
    if not os.path.exists(file_path):
        initialize_netcdf(file_path)
    
    with Dataset(file_path, 'a') as root_grp:
        size_var = root_grp.variables['size']
        loc_x_var = root_grp.variables['location_x']
        loc_y_var = root_grp.variables['location_y']
        loc_z_var = root_grp.variables['location_z']
        cloud_points_var = root_grp.variables['cloud_points']

        # Write data for active clouds at this timestep
        active_tracks = list(tracks.keys())
        for i, track_id in enumerate(active_tracks):
            cloud = tracks[track_id][-1]  # Get the last cloud which represents the current state
            if cloud.is_active:
                size_var[i, timestep] = cloud.size
                loc_x_var[i, timestep], loc_y_var[i, timestep], loc_z_var[i, timestep] = cloud.location
                points = np.array([list(p) for p in cloud.points[:10000]])
                cloud_points_var[i, timestep, :len(points), :] = points
            else:
                # Set current and future entries to NaN for inactive clouds
                size_var[i, timestep:] = np.nan
                loc_x_var[i, timestep:] = np.nan
                loc_y_var[i, timestep:] = np.nan
                loc_z_var[i, timestep:] = np.nan
                cloud_points_var[i, timestep:, :, :] = np.nan

