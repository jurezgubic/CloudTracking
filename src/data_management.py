from netCDF4 import Dataset
import numpy as np
from lib.cloudfield import CloudField
# from memory_profiler import profile # Keep if you use it
import gc

# @profile
def load_cloud_field_from_file(lba_file_path, timestep_idx, config): # MODIFIED signature
    """
    Load cloud data from a single LBA file for a specific conceptual timestep
    and create a CloudField object.
    
    Returns:
    - A CloudField object for the given timestep, or None if loading fails.
    """
    print(f"Loading LBA data from: {lba_file_path} for conceptual timestep {timestep_idx}...")
    
    var_map = config['lba_var_map']
    coord_names = config['lba_coord_names']
    horizontal_resolution = config['horizontal_resolution']

    try:
        with Dataset(lba_file_path, 'r') as dataset:
            # Load or generate grid coordinates
            try:
                xt = dataset.variables[coord_names['x']][:]
            except KeyError:
                print(f"Coordinate variable '{coord_names['x']}' not found. Generating from dimension size and resolution.")
                if 'x' in dataset.dimensions:
                    dim_x_size = dataset.dimensions['x'].size
                    xt = np.arange(dim_x_size) * horizontal_resolution
                else:
                    raise KeyError(f"Dimension 'x' not found in LBA file: {lba_file_path} for generating coordinates.")
            
            try:
                yt = dataset.variables[coord_names['y']][:]
            except KeyError:
                print(f"Coordinate variable '{coord_names['y']}' not found. Generating from dimension size and resolution.")
                if 'y' in dataset.dimensions:
                    dim_y_size = dataset.dimensions['y'].size
                    yt = np.arange(dim_y_size) * horizontal_resolution
                else:
                    raise KeyError(f"Dimension 'y' not found in LBA file: {lba_file_path} for generating coordinates.")

            # Load z coordinate - assuming 'z' variable exists as per previous discussions
            zt = dataset.variables[coord_names['z']][:] 
            
            # Load data variables. LBA files have a time dimension of 1, so access with [0, ...]
            l_data_raw = dataset.variables[var_map['l']][0, :, :, :]
            w_data_raw = dataset.variables[var_map['w']][0, :, :, :]
            p_data_raw = dataset.variables[var_map['p']][0, :, :, :]
            theta_l_data_raw = dataset.variables[var_map['t']][0, :, :, :]
            
            # Calculate q_t_data = q_cloud_liquid_mass + q_ice_mass
            q_ice_data_raw = dataset.variables[var_map['q_ice']][0, :, :, :]
            q_t_data_raw = l_data_raw + q_ice_data_raw
            
            # Transpose data from (x, y, z) to (z, y, x) for internal processing
            # Original axes: 0=x, 1=y, 2=z
            # Target axes:   0=z, 1=y, 2=x
            l_data = l_data_raw.transpose(2, 1, 0)
            w_data = w_data_raw.transpose(2, 1, 0)
            p_data = p_data_raw.transpose(2, 1, 0)
            theta_l_data = theta_l_data_raw.transpose(2, 1, 0)
            q_t_data = q_t_data_raw.transpose(2, 1, 0)
            
            # Ensure data is not masked array, convert to simple numpy array if necessary
            if hasattr(l_data, 'filled'): l_data = l_data.filled(np.nan)
            if hasattr(w_data, 'filled'): w_data = w_data.filled(np.nan)
            if hasattr(p_data, 'filled'): p_data = p_data.filled(np.nan)
            if hasattr(theta_l_data, 'filled'): theta_l_data = theta_l_data.filled(np.nan)
            if hasattr(q_t_data, 'filled'): q_t_data = q_t_data.filled(np.nan)

    except FileNotFoundError:
        print(f"Error: LBA file not found: {lba_file_path}")
        return None
    except KeyError as e:
        print(f"Error: Variable or Dimension {e} not found in LBA file: {lba_file_path}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {lba_file_path}: {e}")
        return None

    # Create a CloudField object
    cloud_field = CloudField(l_data, w_data, p_data, theta_l_data, q_t_data, 
                             timestep_idx, config, xt, yt, zt)
    
    del l_data, w_data, p_data, theta_l_data, q_t_data
    del l_data_raw, w_data_raw, p_data_raw, theta_l_data_raw, q_ice_data_raw, q_t_data_raw # Clear raw versions too
    gc.collect()
    
    return cloud_field

def load_u_field(lba_file_path, config): # MODIFIED signature
    """Load u wind data from a single LBA file."""
    try:
        with Dataset(lba_file_path, 'r') as dataset:
            u_data = dataset.variables[config['lba_var_map']['u']][0, :, :, :]
            if hasattr(u_data, 'filled'): u_data = u_data.filled(np.nan)
            return u_data
    except Exception as e:
        print(f"Error loading u_field from {lba_file_path}: {e}")
        return None


def load_v_field(lba_file_path, config): # MODIFIED signature
    """Load v wind data from a single LBA file."""
    try:
        with Dataset(lba_file_path, 'r') as dataset:
            v_data = dataset.variables[config['lba_var_map']['v']][0, :, :, :]
            if hasattr(v_data, 'filled'): v_data = v_data.filled(np.nan)
            return v_data
    except Exception as e:
        print(f"Error loading v_field from {lba_file_path}: {e}")
        return None

def load_w_field(lba_file_path, config): # MODIFIED signature
    """Load w wind data from a single LBA file."""
    try:
        with Dataset(lba_file_path, 'r') as dataset:
            w_data = dataset.variables[config['lba_var_map']['w']][0, :, :, :]
            if hasattr(w_data, 'filled'): w_data = w_data.filled(np.nan)
            return w_data
    except Exception as e:
        print(f"Error loading w_field from {lba_file_path}: {e}")
        return None

# These functions below might become fully redundant if all necessary data for CloudField
# is loaded within load_cloud_field_from_file. Review if they are still needed.
# For now, they are commented out as their LBA equivalents are not explicitly defined/used elsewhere yet.

# def load_p_field(lba_file_path, config):
#     """Load pressure data from a single LBA file."""
#     with Dataset(lba_file_path, 'r') as dataset:
#         return dataset.variables[config['lba_var_map']['p']][0, :, :, :]

# def load_theta_l_field(lba_file_path, config):
#     """Load liquid water potential temperature data from a single LBA file."""
#     with Dataset(lba_file_path, 'r') as dataset:
#         return dataset.variables[config['lba_var_map']['t']][0, :, :, :]

# def load_q_t_field(lba_file_path, config): # This would need to compute sum as in load_cloud_field
#     """Load total water mixing ratio data from a single LBA file."""
#     # This would be more complex, needing to sum components.
#     # It's better handled in load_cloud_field_from_file directly.
#     pass


def calculate_mean_velocities(lba_file_path, config): # MODIFIED signature
    """Calculates mean wind velocities for each z-level from the given LBA timestep file."""
    u_data = load_u_field(lba_file_path, config)
    v_data = load_v_field(lba_file_path, config)
    
    if u_data is None or v_data is None:
        print(f"Warning: Could not load U or V data from {lba_file_path}. Returning NaN for mean velocities.")
        # Determine expected number of z levels, e.g. by trying to load zt if not otherwise available
        # This is a placeholder; robustly getting z_levels_count might need another way if u/v failed early
        try:
            with Dataset(lba_file_path, 'r') as ds_temp:
                z_levels_count = len(ds_temp.variables[config['lba_coord_names']['z']])
        except:
            z_levels_count = 1 # Fallback, adjust as needed
        return np.full(z_levels_count, np.nan), np.full(z_levels_count, np.nan)

    mean_u = np.mean(u_data, axis=(1, 2))
    mean_v = np.mean(v_data, axis=(1, 2))
    return mean_u, mean_v


