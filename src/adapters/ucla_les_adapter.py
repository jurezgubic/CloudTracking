"""
UCLA-LES / DALES data adapter.

Handles the file format used by UCLA-LES and DALES models where:
- Each variable is in a separate file (e.g., rico.l.nc, rico.u.nc)
- All timesteps are in one file per variable
- Dimension order is (time, z, y, x)
- theta_l is directly output (not perturbation)
- Pressure is full pressure in Pa
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from netCDF4 import Dataset
import gc

from .base_adapter import BaseDataAdapter


class UCLALESAdapter(BaseDataAdapter):
    """
    Data adapter for UCLA-LES / DALES output format.
    
    Expected file structure:
        base_path/
            {prefix}.l.nc  - liquid water mixing ratio
            {prefix}.u.nc  - zonal wind
            {prefix}.v.nc  - meridional wind
            {prefix}.w.nc  - vertical velocity
            {prefix}.p.nc  - pressure
            {prefix}.t.nc  - liquid water potential temperature
            {prefix}.q.nc  - total water mixing ratio
            {prefix}.r.nc  - rain water mixing ratio (optional)
    
    Configuration keys:
        'base_file_path': str - Path to directory containing data files
        'file_name': dict - Mapping of variable names to file names
            Required keys: 'l', 'u', 'v', 'w', 'p', 't', 'q'
            Optional keys: 'r'
    
    Example:
        config = {
            'data_format': 'UCLA-LES',
            'base_file_path': '/path/to/data/',
            'file_name': {
                'l': 'rico.l.nc',
                'u': 'rico.u.nc',
                ...
            }
        }
        adapter = UCLALESAdapter(config)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the UCLA-LES adapter.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with 'base_file_path' and 'file_name'
        """
        super().__init__(config)
        
        self.base_path = Path(config['base_file_path'])
        self.file_names = config['file_name']
        
        # Validate required files exist
        required_vars = ['l', 'u', 'v', 'w', 'p', 't', 'q']
        for var in required_vars:
            if var not in self.file_names:
                raise ValueError(f"Missing required file mapping for variable: {var}")
            
            file_path = self.base_path / self.file_names[var]
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Cache grid info
        self._grid_info: Optional[Dict[str, Any]] = None
        self._n_timesteps: Optional[int] = None
        self._timestep_times: Optional[np.ndarray] = None
    
    def get_total_timesteps(self) -> int:
        """Get the total number of timesteps available."""
        if self._n_timesteps is None:
            # Read from any file (use 'l' as reference)
            file_path = self.base_path / self.file_names['l']
            with Dataset(file_path, 'r') as ds:
                # Time is first dimension
                time_var = ds.variables['time'] if 'time' in ds.variables else None
                if time_var is not None:
                    self._n_timesteps = len(time_var)
                    self._timestep_times = time_var[:]
                else:
                    # Infer from first variable shape
                    l_var = ds.variables['l']
                    self._n_timesteps = l_var.shape[0]
                    self._timestep_times = np.arange(self._n_timesteps)
        
        return self._n_timesteps
    
    def get_timestep_times(self) -> np.ndarray:
        """Get simulation times for each timestep."""
        if self._timestep_times is None:
            self.get_total_timesteps()  # This populates _timestep_times
        return self._timestep_times
    
    def get_grid_info(self) -> Dict[str, Any]:
        """Get grid information from the data files."""
        if self._grid_info is not None:
            return self._grid_info
        
        file_path = self.base_path / self.file_names['l']
        with Dataset(file_path, 'r') as ds:
            xt = ds.variables['xt'][:]
            yt = ds.variables['yt'][:]
            zt = ds.variables['zt'][:]
        
        # Compute grid spacing
        dx = np.median(np.diff(xt)) if len(xt) > 1 else xt[0]
        dy = np.median(np.diff(yt)) if len(yt) > 1 else yt[0]
        dz = np.diff(zt) if len(zt) > 1 else np.array([zt[0]])
        
        self._grid_info = {
            'dx': float(dx),
            'dy': float(dy),
            'dz': dz,
            'nx': len(xt),
            'ny': len(yt),
            'nz': len(zt),
            'xt': np.asarray(xt),
            'yt': np.asarray(yt),
            'zt': np.asarray(zt),
        }
        
        return self._grid_info
    
    def load_timestep(self, timestep: int) -> Dict[str, np.ndarray]:
        """
        Load all fields for a single timestep.
        
        Parameters
        ----------
        timestep : int
            Timestep index (0-based)
            
        Returns
        -------
        dict
            Dictionary with all required fields in CloudTracker format
        """
        print(f"Loading UCLA-LES data for timestep {timestep}...")
        
        # Get grid coordinates
        grid = self.get_grid_info()
        xt, yt, zt = grid['xt'], grid['yt'], grid['zt']
        
        # Load each variable
        # UCLA-LES format: (time, z, y, x) - no transpose needed
        
        with Dataset(self.base_path / self.file_names['l'], 'r') as ds:
            l_data = np.asarray(ds.variables['l'][timestep, :, :, :])
        
        with Dataset(self.base_path / self.file_names['u'], 'r') as ds:
            u_data = np.asarray(ds.variables['u'][timestep, :, :, :])
        
        with Dataset(self.base_path / self.file_names['v'], 'r') as ds:
            v_data = np.asarray(ds.variables['v'][timestep, :, :, :])
        
        with Dataset(self.base_path / self.file_names['w'], 'r') as ds:
            w_data = np.asarray(ds.variables['w'][timestep, :, :, :])
        
        with Dataset(self.base_path / self.file_names['p'], 'r') as ds:
            p_data = np.asarray(ds.variables['p'][timestep, :, :, :])
        
        with Dataset(self.base_path / self.file_names['t'], 'r') as ds:
            theta_l_data = np.asarray(ds.variables['t'][timestep, :, :, :])
        
        with Dataset(self.base_path / self.file_names['q'], 'r') as ds:
            q_t_data = np.asarray(ds.variables['q'][timestep, :, :, :])
        
        # Rain is optional
        if 'r' in self.file_names:
            r_file = self.base_path / self.file_names['r']
            if r_file.exists():
                try:
                    with Dataset(r_file, 'r') as ds:
                        r_data = np.asarray(ds.variables['r'][timestep, :, :, :])
                except (KeyError, OSError):
                    print(f"Could not load rain data, using zeros")
                    r_data = np.zeros_like(l_data)
            else:
                r_data = np.zeros_like(l_data)
        else:
            r_data = np.zeros_like(l_data)
        
        data = {
            'l': l_data,
            'u': u_data,
            'v': v_data,
            'w': w_data,
            'p': p_data,
            'theta_l': theta_l_data,
            'q_t': q_t_data,
            'r': r_data,
            'xt': xt,
            'yt': yt,
            'zt': zt,
        }
        
        gc.collect()
        
        return data
    
    def load_mean_velocities(self, timestep: int) -> tuple:
        """
        Load horizontally-averaged wind profiles efficiently.
        
        Only loads u and v fields instead of all fields.
        """
        with Dataset(self.base_path / self.file_names['u'], 'r') as ds:
            u_data = np.asarray(ds.variables['u'][timestep, :, :, :])
        
        with Dataset(self.base_path / self.file_names['v'], 'r') as ds:
            v_data = np.asarray(ds.variables['v'][timestep, :, :, :])
        
        mean_u = np.mean(u_data, axis=(1, 2))  # Average over y, x
        mean_v = np.mean(v_data, axis=(1, 2))
        
        del u_data, v_data
        gc.collect()
        
        return mean_u, mean_v
