from netCDF4 import Dataset
import numpy as np
from lib.cloudfield import CloudField
from memory_profiler import profile
import gc
from typing import Dict, Any, Optional, Union

# Import adapters
from src.adapters.base_adapter import BaseDataAdapter
from src.adapters.ucla_les_adapter import UCLALESAdapter
from src.adapters.monc_adapter import MONCAdapter


# =============================================================================
# Adapter-based data loading (recommended for new code)
# =============================================================================

def create_data_adapter(config: Dict[str, Any]) -> BaseDataAdapter:
    """
    Factory function to create the appropriate data adapter based on config.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary. Must contain 'data_format' key with value
        'UCLA-LES' or 'MONC'.
        
        For UCLA-LES:
            - 'base_file_path': str - Path to data directory
            - 'file_name': dict - Mapping of variables to filenames
            
        For MONC:
            - 'monc_data_path': str - Path to MONC output directory
            - 'monc_config_file': str - Path to .mcf config file
            
    Returns
    -------
    BaseDataAdapter
        Configured data adapter instance
        
    Raises
    ------
    ValueError
        If data_format is not recognized
    """
    data_format = config.get('data_format', 'UCLA-LES')
    
    if data_format.upper() in ['UCLA-LES', 'UCLA_LES', 'DALES', 'RICO']:
        return UCLALESAdapter(config)
    elif data_format.upper() == 'MONC':
        return MONCAdapter(config)
    else:
        raise ValueError(
            f"Unknown data format: {data_format}. "
            f"Supported formats: 'UCLA-LES', 'MONC'"
        )


def load_cloud_field_from_adapter(
    adapter: BaseDataAdapter,
    timestep: int,
    config: Dict[str, Any]
) -> CloudField:
    """
    Load cloud data using an adapter and create a CloudField object.
    
    This is the recommended way to load data as it supports multiple formats.
    
    Parameters
    ----------
    adapter : BaseDataAdapter
        Data adapter instance (created via create_data_adapter)
    timestep : int
        Timestep index to load (0-based)
    config : dict
        CloudTracker configuration dictionary
        
    Returns
    -------
    CloudField
        CloudField object for the given timestep
    """
    # Load data using adapter
    data = adapter.load_timestep(timestep)
    
    # Create CloudField object
    cloud_field = CloudField(
        data['l'],
        data['u'],
        data['v'],
        data['w'],
        data['p'],
        data['theta_l'],
        data['q_t'],
        data['r'],
        timestep,
        config,
        data['xt'],
        data['yt'],
        data['zt']
    )
    
    gc.collect()
    
    return cloud_field


def calculate_mean_velocities_from_adapter(
    adapter: BaseDataAdapter,
    timestep: int
) -> tuple:
    """
    Calculate mean velocities using an adapter.
    
    Parameters
    ----------
    adapter : BaseDataAdapter
        Data adapter instance
    timestep : int
        Timestep index
        
    Returns
    -------
    tuple
        (mean_u, mean_v) arrays of shape (nz,)
    """
    return adapter.load_mean_velocities(timestep)


# =============================================================================
# Legacy functions (for backward compatibility with existing code)
# =============================================================================

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
    
    with Dataset(f"{file_path}{file_names['u']}", 'r') as dataset:
        u_data = dataset.variables['u'][timestep, :, :, :]

    with Dataset(f"{file_path}{file_names['v']}", 'r') as dataset:
        v_data = dataset.variables['v'][timestep, :, :, :]

    with Dataset(f"{file_path}{file_names['p']}", 'r') as dataset:
        p_data = dataset.variables['p'][timestep, :, :, :]
    
    with Dataset(f"{file_path}{file_names['t']}", 'r') as dataset:
        theta_l_data = dataset.variables['t'][timestep, :, :, :]
    
    with Dataset(f"{file_path}{file_names['q']}", 'r') as dataset:
        q_t_data = dataset.variables['q'][timestep, :, :, :]
    
    # Load 'w' data regardless of w_switch (needed for calculations)
    with Dataset(f"{file_path}{file_names['w']}", 'r') as dataset:
        w_data = dataset.variables['w'][timestep, :, :, :]
    
    # Load rain water mixing ratio (optional - zeros if file missing)
    # Note: Rain is excluded from saturation adjustment but included in density loading
    if 'r' in file_names:
        r_file_path = f"{file_path}{file_names['r']}"
        try:
            with Dataset(r_file_path, 'r') as dataset:
                r_data = dataset.variables['r'][timestep, :, :, :]
        except (FileNotFoundError, OSError):
            print(f"Rain file not found at {r_file_path}, using zeros for rain water.")
            r_data = np.zeros_like(l_data)
    else:
        r_data = np.zeros_like(l_data)
    
    # Create a CloudField object
    cloud_field = CloudField(l_data, u_data, v_data, w_data, p_data, theta_l_data, q_t_data, r_data, timestep, config, xt, yt, zt)
    
    # Explicitly clear variables to help garbage collection
    del l_data, u_data, v_data, p_data, theta_l_data, q_t_data, w_data, r_data
    
    gc.collect()  # Force garbage collection
    
    return cloud_field

def load_u_field(file_path, file_name, timestep):
    """
    Load u wind data from files for a specific timestep.
    Returns: u wind data for the given timestep.
    """
    with Dataset(f"{file_path}{file_name}", 'r') as u_dataset:
        return np.array(u_dataset.variables['u'][timestep, :, :, :])

def load_v_field(file_path, file_name, timestep):
    """
    Load v wind data from files for a specific timestep.
    Returns: v wind data for the given timestep.
    """
    with Dataset(f"{file_path}{file_name}", 'r') as v_dataset:
        return np.array(v_dataset.variables['v'][timestep, :, :, :])

# currently redundant but will be used to get rid of load_cloud_field_from_file
# which will be split into load_w_field and load_l_field
def load_w_field(file_path, file_name, timestep):
    """
    Load w wind data from files for a specific timestep.
    Returns: w wind data for the given timestep.
    """
    with Dataset(f"{file_path}{file_name}", 'r') as w_dataset:
        return np.array(w_dataset.variables['w'][timestep, :, :, :])

# currently redundant but will be used to get rid of load_cloud_field_from_file
def load_p_field(file_path, file_name, timestep):
    """
    Load pressure data from files for a specific timestep.
    Returns: pressure data for the given timestep.
    """
    with Dataset(f"{file_path}{file_name}", 'r') as p_dataset:
        return np.array(p_dataset.variables['p'][timestep, :, :, :])

# currently redundant but will be used to get rid of load_cloud_field_from_file
def load_theta_l_field(file_path, file_name, timestep):
    """
    Load liq water potential temperature data from files for a specific timestep.
    Returns: liq watwe potential temperature data for the given timestep.
    """
    with Dataset(f"{file_path}{file_name}", 'r') as theta_l_dataset:
        return np.array(theta_l_dataset.variables['theta_l'][timestep, :, :, :])

# currently redundant but will be used to get rid of load_cloud_field_from_file
def load_q_t_field(file_path, file_name, timestep):
    """
    Load total water mixing ratio data from files for a specific timestep.
    Returns: total water mixing ratio data for the given timestep.
    """
    with Dataset(f"{file_path}{file_name}", 'r') as q_t_dataset:
        return np.array(q_t_dataset.variables['q_t'][timestep, :, :, :])

def calculate_mean_velocities(file_path, file_names, timestep):
    """Calculates mean wind velocities for each z-level at the given timestep."""
    u_data = load_u_field(file_path, file_names['u'], timestep)
    v_data = load_v_field(file_path, file_names['v'], timestep)
    mean_u = np.mean(u_data, axis=(1, 2))  # Average over y and x dimensions
    mean_v = np.mean(v_data, axis=(1, 2))
    return mean_u, mean_v


