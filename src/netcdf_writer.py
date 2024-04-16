import xarray as xr
import numpy as np

def write_cloud_tracks_to_netcdf(tracks, file_path):
    """
    Write cloud track information to a netCDF file using xarray.

    Parameters:
    - tracks: Dictionary containing cloud track information, including points for each cloud.
    - file_path: Path to the netCDF file to be created.
    """

    # Prepare data arrays
    max_length = max(len(track) for track in tracks.values())
    num_tracks = len(tracks)

    # Initializing data arrays with NaNs
    sizes = np.full((num_tracks, max_length), np.nan, dtype=np.float32)
    locations_x = np.full((num_tracks, max_length), np.nan, dtype=np.float32)
    locations_y = np.full((num_tracks, max_length), np.nan, dtype=np.float32)
    locations_z = np.full((num_tracks, max_length), np.nan, dtype=np.float32)

    # Prepare a variable for cloud points, initializing with a fixed size or dynamically adjust based on maximum points found
    # Example with fixed size, adjust `max_points` as needed
    max_points = 10000  # You might want to calculate this based on your data
    cloud_points = np.full((num_tracks, max_length, max_points, 3), np.nan, dtype=np.float32)



    # Fill data arrays
    track_ids = list(tracks.keys())
    for i, track_id in enumerate(track_ids):
        for j, cloud in enumerate(tracks[track_id]):
            sizes[i, j] = cloud.size
            locations_x[i, j], locations_y[i, j], locations_z[i, j] = cloud.location
            # For cloud points, assume cloud.points is a list of tuples [(x1, y1), (x2, y2), ...]
            for k, point in enumerate(cloud.points[:max_points]):
                cloud_points[i, j, k, :] = point

    # Create xarray Dataset
    ds = xr.Dataset({
        "size": (["track", "time"], sizes),
        "location_x": (["track", "time"], locations_x),
        "location_y": (["track", "time"], locations_y),
        "location_z": (["track", "time"], locations_z),
        "cloud_points": (["track", "time", "point", "coordinate"], cloud_points),
    }, coords={
        "track": track_ids,
        "time": np.arange(max_length),
        "point": np.arange(max_points),
        "coordinate": np.arange(3)  # Ensure this aligns with your data structure for 3D points
    })

    # Add global attributes
    ds.attrs["description"] = "Cloud tracking information"

    # Write to netCDF
    ds.to_netcdf(file_path)

