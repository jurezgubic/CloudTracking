import numpy as np
from netCDF4 import Dataset

def write_cloud_data_to_netcdf(cloud_tracks, filename):
    with Dataset(filename, 'w', format='NETCDF4') as dataset:
        # Create dimensions
        max_timesteps = max(len(track) for track in cloud_tracks.values())
        max_clouds = len(cloud_tracks)
        max_points = max(len(cloud.points) for track in cloud_tracks.values() for cloud in track)

        dataset.createDimension('timestep', max_timesteps)
        dataset.createDimension('cloud', max_clouds)
        dataset.createDimension('point', max_points)

        # Create variables
        sizes = dataset.createVariable('size', np.float32, ('cloud', 'timestep'))
        x_centers = dataset.createVariable('x_center', np.float32, ('cloud', 'timestep'))
        y_centers = dataset.createVariable('y_center', np.float32, ('cloud', 'timestep'))
        z_centers = dataset.createVariable('z_center', np.float32, ('cloud', 'timestep'))
        x_coords = dataset.createVariable('x', np.float32, ('cloud', 'timestep', 'point'))
        y_coords = dataset.createVariable('y', np.float32, ('cloud', 'timestep', 'point'))
        z_coords = dataset.createVariable('z', np.float32, ('cloud', 'timestep', 'point'))

        # Store data
        for i, (cloud_id, track) in enumerate(cloud_tracks.items()):
            for j, cloud in enumerate(track):
                sizes[i, j] = cloud.size
                x_centers[i, j], y_centers[i, j], z_centers[i, j] = cloud.location
                for k, point in enumerate(cloud.points):
                    x_coords[i, j, k], y_coords[i, j, k], z_coords[i, j, k] = point

        # Handle missing data
        x_coords[:, :, :] = np.ma.masked_where(x_coords[:, :, :] == 0, x_coords)
        y_coords[:, :, :] = np.ma.masked_where(y_coords[:, :, :] == 0, y_coords)
        z_coords[:, :, :] = np.ma.masked_where(z_coords[:, :, :] == 0, z_coords)

