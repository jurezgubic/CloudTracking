from netCDF4 import Dataset
import numpy as np
from lib.cloudfield import CloudField
from memory_profiler import profile

# @profile
def load_cloud_field_from_file(file_path, file_names, timestep, config):
    """
    Load cloud data from files for a specific timestep and create a CloudField object.

    Returns:
    - A CloudField object for the given timestep.
    """
    # Load 'l' data
    l_dataset = Dataset(f"{file_path}{file_names['l']}", 'r')
    l_data = l_dataset.variables['l'][timestep, :, :, :]

    # Load 'w' data if vertical velocity condition is enabled
    w_data = None
    if config['w_switch']:
        w_dataset = Dataset(f"{file_path}{file_names['w']}", 'r')
        w_data = w_dataset.variables['w'][timestep, :, :, :]

    # Load grid coordinates
    xt = l_dataset.variables['xt'][:]
    yt = l_dataset.variables['yt'][:]
    zt = l_dataset.variables['zt'][:]

    # Create a CloudField object
    cloud_field = CloudField(l_data, w_data, timestep, config, xt, yt, zt)

    # Memory management
    del l_data, xt, yt, zt
    if w_data is not None:
        del w_data

    return cloud_field

def load_u_field(file_path, file_name, timestep):
    """
    Load u wind data from files for a specific timestep.
    Returns: u wind data for the given timestep.
    """
    u_dataset = Dataset(f"{file_path}{file_name}", 'r')
    return u_dataset.variables['u'][timestep, :, :, :]

def load_v_field(file_path, file_name, timestep):
    """
    Load v wind data from files for a specific timestep.
    Returns: v wind data for the given timestep.
    """
    v_dataset = Dataset(f"{file_path}{file_name}", 'r')
    return v_dataset.variables['v'][timestep, :, :, :] 

# currently redundant but will be used to get rid of load_cloud_field_from_file
# which will be split into load_w_field and load_l_field
def load_w_field(file_path, file_name, timestep):
    """
    Load w wind data from files for a specific timestep.
    Returns: w wind data for the given timestep.
    """
    w_dataset = Dataset(f"{file_path}{file_name}", 'r')
    return w_dataset.variables['w'][timestep, :, :, :]


def calculate_mean_velocities(file_path, file_names, timestep):
    """Calculates mean wind velocities for each z-level at the given timestep."""
    u_data = load_u_field(file_path, file_names['u'], timestep)
    v_data = load_v_field(file_path, file_names['v'], timestep)
    mean_u = np.mean(u_data, axis=(1, 2))  # Average over y and x dimensions
    mean_v = np.mean(v_data, axis=(1, 2))
    return mean_u, mean_v


