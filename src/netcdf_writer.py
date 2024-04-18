from netCDF4 import Dataset
import numpy as np
import os

def write_cloud_tracks_to_netcdf(tracks, file_path, timestep):
    # Ensure the file exists and open for appending or create if not exists
    if not os.path.exists(file_path):
        root_grp = Dataset(file_path, 'w', format='NETCDF4')
        root_grp.createDimension('track', None)  # Unlimited dimension
        root_grp.createDimension('time', None)  # Unlimited dimension
        root_grp.createDimension('point', 10000)  # Static dimension
        root_grp.createDimension('coordinate', 3)  # Static dimension for 3D points

        root_grp.createVariable('size', 'f4', ('track', 'time'))
        root_grp.createVariable('location_x', 'f4', ('track', 'time'))
        root_grp.createVariable('location_y', 'f4', ('track', 'time'))
        root_grp.createVariable('location_z', 'f4', ('track', 'time'))
        root_grp.createVariable('cloud_points', 'f4', ('track', 'time', 'point', 'coordinate'))

        root_grp.close()

    # Open the file for appending
    with Dataset(file_path, 'a') as root_grp:
        # Check if the variables exist (important if the file was just created)
        if 'size' not in root_grp.variables:
            size_var = root_grp.createVariable('size', 'f4', ('track', 'time'))
        else:
            size_var = root_grp.variables['size']

        if 'location_x' not in root_grp.variables:
            loc_x_var = root_grp.createVariable('location_x', 'f4', ('track', 'time'))
        else:
            loc_x_var = root_grp.variables['location_x']

        if 'location_y' not in root_grp.variables:
            loc_y_var = root_grp.createVariable('location_y', 'f4', ('track', 'time'))
        else:
            loc_y_var = root_grp.variables['location_y']

        if 'location_z' not in root_grp.variables:
            loc_z_var = root_grp.createVariable('location_z', 'f4', ('track', 'time'))
        else:
            loc_z_var = root_grp.variables['location_z']

        if 'cloud_points' not in root_grp.variables:
            cloud_points_var = root_grp.createVariable('cloud_points', 'f4', ('track', 'time', 'point', 'coordinate'))
        else:
            cloud_points_var = root_grp.variables['cloud_points']

        # Make sure we have enough space for tracks and timesteps
        num_tracks = len(tracks)
        required_tracks = max(num_tracks, len(root_grp.dimensions['track']))
        required_timesteps = timestep + 1  # Assuming timestep is zero-indexed

        # Assign data for each track
        for i, track_id in enumerate(tracks):
            track = tracks[track_id][-1]  # Assuming the latest state is what you want
            size_var[i, timestep] = track.size
            loc_x_var[i, timestep], loc_y_var[i, timestep], loc_z_var[i, timestep] = track.location
            points = np.array([list(p) for p in track.points[:10000]])
            if len(points) > 0:  # Ensure there are points to record
                cloud_points_var[i, timestep, :len(points), :] = points

