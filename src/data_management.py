from netCDF4 import Dataset
import numpy as np
from lib.cloudfield import CloudField


def load_cloud_field_from_file(file_path, file_name, timestep, config):
    """
    Load cloud data from files for a specific timestep and create a CloudField object.

    Returns:
    - A CloudField object for the given timestep.
    """
    # Load 'l' data
    l_dataset = Dataset(f"{file_path}{file_name['l']}", 'r')
    l_data = l_dataset.variables['l'][timestep, :, :, :]
    xt = l_dataset.variables['xt'][:]
    yt = l_dataset.variables['yt'][:]
    zt = l_dataset.variables['zt'][:]

    # Load 'u' data
    u_dataset = Dataset(f"{file_path}{file_name['u']}", 'r')
    u_data = u_dataset.variables['u'][timestep, :, :, :]

    # Load 'v' data
    v_dataset = Dataset(f"{file_path}{file_name['v']}", 'r')
    v_data = v_dataset.variables['v'][timestep, :, :, :]

    # Load 'w' data
    w_dataset = Dataset(f"{file_path}{file_name['w']}", 'r')
    w_data = w_dataset.variables['w'][timestep, :, :, :]

    return CloudField(l_data, timestep, config, xt, yt, zt)

