from netCDF4 import Dataset
import numpy as np

def write_cloud_tracks_to_netcdf(tracks, file_path):
    """
    Write cloud track information to a netCDF file.

    Parameters:
    - tracks: Dictionary containing cloud track information.
    - file_path: Path to the netCDF file to be created.
    """
    with Dataset(file_path, 'w', format='NETCDF4') as nc:
        # Dimensions
        max_length = max(len(track) for track in tracks.values())
        nc.createDimension('track', None)  # Unlimited dimension
        nc.createDimension('time', max_length)

        # Variables
        sizes = nc.createVariable('size', 'f4', ('track', 'time'))
        locations_x = nc.createVariable('location_x', 'f4', ('track', 'time'))
        locations_y = nc.createVariable('location_y', 'f4', ('track', 'time'))
        locations_z = nc.createVariable('location_z', 'f4', ('track', 'time'))

        # Fill variables with data
        track_ids = list(tracks.keys())
        for i, track_id in enumerate(track_ids):
            for j, cloud in enumerate(tracks[track_id]):
                sizes[i, j] = cloud.size
                locations_x[i, j], locations_y[i, j], locations_z[i, j] = cloud.location

        # Attributes
        nc.description = "Cloud tracking information"

