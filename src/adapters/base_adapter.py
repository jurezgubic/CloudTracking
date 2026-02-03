"""
Abstract base class for LES data adapters.

This module defines the interface that all data adapters must implement
to provide data to CloudTracker in a consistent format.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class BaseDataAdapter(ABC):
    """
    Abstract base class for LES data adapters.
    
    All adapters must implement methods to load data and provide it in the format
    expected by CloudField:
    - Arrays in (z, y, x) dimension order
    - Water species in kg/kg
    - Pressure in Pa
    - Temperature as liquid water potential temperature (theta_l) in K
    - Coordinates in meters
    
    Attributes
    ----------
    config : dict
        CloudTracker configuration dictionary
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the adapter.
        
        Parameters
        ----------
        config : dict
            CloudTracker configuration dictionary containing paths and settings
        """
        self.config = config
    
    @abstractmethod
    def get_total_timesteps(self) -> int:
        """
        Get the total number of available timesteps.
        
        Returns
        -------
        int
            Number of timesteps available in the dataset
        """
        pass
    
    @abstractmethod
    def get_timestep_times(self) -> np.ndarray:
        """
        Get the simulation times for each timestep.
        
        Returns
        -------
        np.ndarray
            Array of simulation times in seconds
        """
        pass
    
    @abstractmethod
    def load_timestep(self, timestep: int) -> Dict[str, np.ndarray]:
        """
        Load all required fields for a single timestep.
        
        Parameters
        ----------
        timestep : int
            Timestep index to load (0-based)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'l': Cloud liquid water mixing ratio (z, y, x) [kg/kg]
            - 'u': Zonal wind (z, y, x) [m/s]
            - 'v': Meridional wind (z, y, x) [m/s]
            - 'w': Vertical velocity (z, y, x) [m/s]
            - 'p': Pressure (z, y, x) [Pa]
            - 'theta_l': Liquid water potential temperature (z, y, x) [K]
            - 'q_t': Total water mixing ratio (z, y, x) [kg/kg]
            - 'r': Rain water mixing ratio (z, y, x) [kg/kg]
            - 'xt': X coordinates (nx,) [m]
            - 'yt': Y coordinates (ny,) [m]
            - 'zt': Z coordinates (nz,) [m]
        """
        pass
    
    @abstractmethod
    def get_grid_info(self) -> Dict[str, Any]:
        """
        Get grid information.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'dx': Horizontal grid spacing in x [m]
            - 'dy': Horizontal grid spacing in y [m]
            - 'dz': Vertical grid spacing array or scalar [m]
            - 'nx': Number of grid points in x
            - 'ny': Number of grid points in y
            - 'nz': Number of grid points in z
            - 'xt': X coordinates [m]
            - 'yt': Y coordinates [m]
            - 'zt': Z coordinates [m]
        """
        pass
    
    def load_mean_velocities(self, timestep: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load horizontally-averaged wind profiles for a timestep.
        
        Default implementation loads full fields and computes mean.
        Subclasses may override for efficiency.
        
        Parameters
        ----------
        timestep : int
            Timestep index
            
        Returns
        -------
        tuple
            (mean_u, mean_v) arrays of shape (nz,)
        """
        data = self.load_timestep(timestep)
        mean_u = np.mean(data['u'], axis=(1, 2))  # Average over y, x
        mean_v = np.mean(data['v'], axis=(1, 2))
        return mean_u, mean_v
    
    def validate_data(self, data: Dict[str, np.ndarray]) -> None:
        """
        Validate loaded data for consistency.
        
        Parameters
        ----------
        data : dict
            Data dictionary from load_timestep
            
        Raises
        ------
        ValueError
            If data is inconsistent or invalid
        """
        required_keys = ['l', 'u', 'v', 'w', 'p', 'theta_l', 'q_t', 'r', 'xt', 'yt', 'zt']
        
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required field: {key}")
        
        # Check 3D field shapes match
        shape_3d = data['l'].shape
        for key in ['u', 'v', 'w', 'p', 'theta_l', 'q_t', 'r']:
            if data[key].shape != shape_3d:
                raise ValueError(
                    f"Shape mismatch: {key} has shape {data[key].shape}, "
                    f"expected {shape_3d}"
                )
        
        # Check coordinate lengths match
        nz, ny, nx = shape_3d
        if len(data['xt']) != nx:
            raise ValueError(f"xt length {len(data['xt'])} doesn't match nx={nx}")
        if len(data['yt']) != ny:
            raise ValueError(f"yt length {len(data['yt'])} doesn't match ny={ny}")
        if len(data['zt']) != nz:
            raise ValueError(f"zt length {len(data['zt'])} doesn't match nz={nz}")
        
        # Check for reasonable value ranges
        if np.any(data['l'] < 0):
            raise ValueError("Negative values in liquid water mixing ratio")
        if np.any(data['q_t'] < 0):
            raise ValueError("Negative values in total water mixing ratio")
        if np.any(data['theta_l'] < 100) or np.any(data['theta_l'] > 1000):
            raise ValueError(
                f"Theta_l values out of reasonable range: "
                f"[{data['theta_l'].min():.1f}, {data['theta_l'].max():.1f}] K"
            )
        if np.any(data['p'] < 1000) or np.any(data['p'] > 150000):
            raise ValueError(
                f"Pressure values out of reasonable range: "
                f"[{data['p'].min():.1f}, {data['p'].max():.1f}] Pa"
            )
