from netCDF4 import Dataset
import numpy as np
from lib.cloudfield import CloudField 

def load_cloud_field_from_file(l_file_path, w_file_path, timestep, config):
    # Load 'l' variable data
    with Dataset(l_file_path, 'r') as nc:
        l_data = nc.variables['l'][timestep, :, :, :]  # Assuming 'l' is the variable name
        l_data = np.array(l_data)  # Ensure it's a numpy array

    # Optionally load 'w' variable data
    w_data = None
    if config['consider_w']:
        with Dataset(w_file_path, 'r') as nc:
            w_data = nc.variables['w'][timestep, :, :, :]  # Assuming 'w' is the variable name
            w_data = np.array(w_data)  # Ensure it's a numpy array

    # Now, l_data and w_data are numpy arrays and can be passed to CloudField
    cloud_field = CloudField(timestep, l_data, w_data, config)
    return cloud_field

