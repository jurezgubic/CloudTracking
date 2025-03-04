from netCDF4 import Dataset
import numpy as np
from lib.cloudfield import CloudField
from memory_profiler import profile
import gc

# @profile
def load_cloud_field_from_file(file_path, file_names, timestep, config):
    """
    Load cloud data from files for a specific timestep and create a CloudField object.
    
    Returns:
    - A CloudField object for the given timestep.
    """
    print(f"Loading data for timestep {timestep}...")
    
    # Load grid coordinates first (needed for all datasets)
    with Dataset(f"{file_path}{file_names['l']}", 'r') as l_dataset:
        xt = l_dataset.variables['xt'][:]
        yt = l_dataset.variables['yt'][:]
        zt = l_dataset.variables['zt'][:]
    
    # Load datasets one by one to minimize memory usage
    with Dataset(f"{file_path}{file_names['l']}", 'r') as dataset:
        l_data = dataset.variables['l'][timestep, :, :, :]
    
    with Dataset(f"{file_path}{file_names['p']}", 'r') as dataset:
        p_data = dataset.variables['p'][timestep, :, :, :]
    
    with Dataset(f"{file_path}{file_names['t']}", 'r') as dataset:
        theta_l_data = dataset.variables['t'][timestep, :, :, :]
    
    with Dataset(f"{file_path}{file_names['q']}", 'r') as dataset:
        q_t_data = dataset.variables['q'][timestep, :, :, :]
    
    # Load 'w' data if vertical velocity condition is enabled
    if config['w_switch']:
        with Dataset(f"{file_path}{file_names['w']}", 'r') as dataset:
            w_data = dataset.variables['w'][timestep, :, :, :]
    else:
        w_data = None
    
    # Create a CloudField object
    cloud_field = CloudField(l_data, w_data, p_data, theta_l_data, q_t_data, timestep, config, xt, yt, zt)
    
    # Explicitly clear variables to help garbage collection
    del l_data, p_data, theta_l_data, q_t_data
    if w_data is not None:
        del w_data
    
    gc.collect()  # Force garbage collection
    
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

# currently redundant but will be used to get rid of load_cloud_field_from_file
def load_p_field(file_path, file_name, timestep):
    """
    Load pressure data from files for a specific timestep.
    Returns: pressure data for the given timestep.
    """
    p_dataset = Dataset(f"{file_path}{file_name}", 'r')
    return p_dataset.variables['p'][timestep, :, :, :]

# currently redundant but will be used to get rid of load_cloud_field_from_file
def load_theta_l_field(file_path, file_name, timestep):
    """
    Load liq water potential temperature data from files for a specific timestep.
    Returns: liq watwe potential temperature data for the given timestep.
    """
    theta_l_dataset = Dataset(f"{file_path}{file_name}", 'r')
    return theta_l_dataset.variables['theta_l'][timestep, :, :, :]

# currently redundant but will be used to get rid of load_cloud_field_from_file
def load_q_t_field(file_path, file_name, timestep):
    """
    Load total water mixing ratio data from files for a specific timestep.
    Returns: total water mixing ratio data for the given timestep.
    """
    q_t_dataset = Dataset(f"{file_path}{file_name}", 'r')
    return q_t_dataset.variables['q_t'][timestep, :, :, :]

def calculate_mean_velocities(file_path, file_names, timestep):
    """Calculates mean wind velocities for each z-level at the given timestep."""
    u_data = load_u_field(file_path, file_names['u'], timestep)
    v_data = load_v_field(file_path, file_names['v'], timestep)
    mean_u = np.mean(u_data, axis=(1, 2))  # Average over y and x dimensions
    mean_v = np.mean(v_data, axis=(1, 2))
    return mean_u, mean_v


