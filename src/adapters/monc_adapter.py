"""
MONC LES data adapter.

Handles the file format used by MONC model where:
- Each timestep is in a separate file (e.g., 3dfields_ts_180.nc)
- All variables are in one file per timestep
- Dimension order is (time, x, y, z) - needs transpose to (z, y, x)
- theta (th) is perturbation from reference profile
- pressure (p) is perturbation pressure
- Staggered grid: w on z levels, scalars on zn levels

Reference state reconstruction:
- theta = th_perturbation + thref(z), where thref is interpolated from config
- pressure is computed hydrostatically from thref profile
- theta_l = theta - (L_v/c_pd) * (q_l/Pi), where Pi is Exner function
"""

import re
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from netCDF4 import Dataset
import gc
import warnings

from .base_adapter import BaseDataAdapter
from ..config_parsers.mcf_parser import MONCConfigParser


# Physical constants (consistent with utils/physics.py)
R_D = 287.04        # Gas constant for dry air [J/kg/K]
R_V = 461.5         # Gas constant for water vapor [J/kg/K]
C_PD = 1005.0       # Specific heat of dry air at constant pressure [J/kg/K]
L_V = 2.5e6         # Latent heat of vaporization [J/kg]
G = 9.81            # Gravitational acceleration [m/s²]


class MONCAdapter(BaseDataAdapter):
    """
    Data adapter for MONC LES output format.
    
    Expected file structure:
        data_path/
            3dfields_ts_180.nc
            3dfields_ts_360.nc
            ...
    
    Configuration keys:
        'data_format': 'MONC'
        'monc_data_path': str - Path to directory containing MONC output files
        'monc_config_file': str - Path to .mcf configuration file
        'monc_file_pattern': str - File pattern (default: '3dfields_ts_{time}.nc')
    
    The .mcf file must contain:
        - z_init_pl_theta, f_init_pl_theta: Reference theta profile
        - dxx, dyy: Horizontal grid spacing
        - surface_pressure: Initial surface pressure [Pa]
        - surface_reference_pressure: Reference pressure p0 [Pa]
    
    Example:
        config = {
            'data_format': 'MONC',
            'monc_data_path': '/path/to/monc/output/',
            'monc_config_file': '/path/to/lba_config.mcf',
        }
        adapter = MONCAdapter(config)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the MONC adapter.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with MONC-specific paths
        """
        super().__init__(config)
        
        # Parse paths
        self.data_path = Path(config['monc_data_path'])
        self.config_file = Path(config['monc_config_file'])
        self.file_pattern = config.get('monc_file_pattern', '3dfields_ts_{time}.nc')
        
        # Validate paths
        if not self.data_path.exists():
            raise FileNotFoundError(f"MONC data path not found: {self.data_path}")
        if not self.config_file.exists():
            raise FileNotFoundError(f"MONC config file not found: {self.config_file}")
        
        # Parse MONC configuration
        print(f"Parsing MONC config: {self.config_file}")
        self.mcf_parser = MONCConfigParser(self.config_file)
        self.mcf_config = self.mcf_parser.parse()
        
        # Get reference profile and pressures
        self.z_thref, self.f_thref = self.mcf_parser.get_reference_theta_profile()
        self.dx, self.dy = self.mcf_parser.get_grid_spacing()
        self.p_surface, self.p0 = self.mcf_parser.get_surface_pressures()
        
        print(f"  dx={self.dx}m, dy={self.dy}m")
        print(f"  p_surface={self.p_surface} Pa, p0={self.p0} Pa")
        print(f"  Reference theta profile: {len(self.z_thref)} points, "
              f"z=[{self.z_thref.min():.0f}, {self.z_thref.max():.0f}] m")
        
        # Discover available files
        self._discover_files()
        
        # Cache grid info (populated on first access)
        self._grid_info: Optional[Dict[str, Any]] = None
        self._thref_on_grid: Optional[np.ndarray] = None
        self._pref_on_grid: Optional[np.ndarray] = None
    
    def _discover_files(self) -> None:
        """Discover available MONC output files and extract timesteps."""
        # Convert pattern to glob pattern
        glob_pattern = self.file_pattern.replace('{time}', '*')
        files = glob.glob(str(self.data_path / glob_pattern))
        
        if not files:
            raise FileNotFoundError(
                f"No MONC output files found matching pattern: "
                f"{self.data_path / glob_pattern}"
            )
        
        # Extract timesteps from filenames
        # Pattern: 3dfields_ts_{time}.nc
        file_time_pairs: List[Tuple[Path, float]] = []
        
        # Build regex from pattern
        regex_pattern = self.file_pattern.replace('{time}', r'(\d+)')
        regex = re.compile(regex_pattern)
        
        for f in files:
            filename = Path(f).name
            match = regex.match(filename)
            if match:
                time_val = float(match.group(1))
                file_time_pairs.append((Path(f), time_val))
        
        # Sort by time (not alphabetically!)
        file_time_pairs.sort(key=lambda x: x[1])
        
        self._files = [p[0] for p in file_time_pairs]
        self._times = np.array([p[1] for p in file_time_pairs])
        
        print(f"  Found {len(self._files)} MONC output files")
        print(f"  Time range: [{self._times.min():.0f}, {self._times.max():.0f}] seconds")
    
    def get_total_timesteps(self) -> int:
        """Get the total number of available timesteps."""
        return len(self._files)
    
    def get_timestep_times(self) -> np.ndarray:
        """Get simulation times for each timestep in seconds."""
        return self._times
    
    def get_grid_info(self) -> Dict[str, Any]:
        """
        Get grid information from the first MONC file.
        
        Note: Vertical grid is read from NetCDF (z, zn arrays),
        horizontal grid is constructed from config (dxx, dyy).
        """
        if self._grid_info is not None:
            return self._grid_info
        
        # Read from first file
        with Dataset(self._files[0], 'r') as ds:
            # Vertical coordinates (in file)
            z = np.asarray(ds.variables['z'][:])    # w levels
            zn = np.asarray(ds.variables['zn'][:])  # scalar levels
            
            # Grid dimensions
            nx = ds.dimensions['x'].size
            ny = ds.dimensions['y'].size
            nz = ds.dimensions['zn'].size  # Use scalar levels for nz
        
        # Construct horizontal coordinates from config
        xt = np.arange(nx) * self.dx + self.dx / 2  # Cell centers
        yt = np.arange(ny) * self.dy + self.dy / 2
        
        # Vertical spacing (varies with height)
        dz = np.diff(zn)
        
        self._grid_info = {
            'dx': self.dx,
            'dy': self.dy,
            'dz': dz,
            'nx': nx,
            'ny': ny,
            'nz': nz,
            'xt': xt,
            'yt': yt,
            'zt': zn,  # Use scalar levels as primary z coordinate
            'z_w': z,   # W levels (staggered)
            'zn': zn,   # Scalar levels
        }
        
        # Precompute reference profiles on grid
        self._compute_reference_profiles(zn)
        
        return self._grid_info
    
    def _compute_reference_profiles(self, zn: np.ndarray) -> None:
        """
        Compute reference theta and pressure profiles on the model grid.
        
        Parameters
        ----------
        zn : np.ndarray
            Scalar level heights [m]
        """
        # Interpolate reference theta to model grid
        # Handle extrapolation for levels outside the defined profile
        self._thref_on_grid = np.interp(zn, self.z_thref, self.f_thref)
        
        # Compute hydrostatic reference pressure
        # Using iterative integration: dp/dz = -rho*g = -p*g/(R_d*T)
        # T = theta * (p/p0)^(R_d/c_pd)
        
        pref = np.zeros_like(zn)
        
        # Start from surface
        # Find first level at or above z=0
        first_above_surface = np.searchsorted(zn, 0)
        if first_above_surface > 0:
            # There are levels below surface (negative z)
            # Extrapolate downward from surface
            pref[first_above_surface] = self.p_surface
            
            # Integrate downward for negative z levels
            for i in range(first_above_surface - 1, -1, -1):
                dz = zn[i+1] - zn[i]  # negative
                theta_avg = 0.5 * (self._thref_on_grid[i] + self._thref_on_grid[i+1])
                T_avg = theta_avg * (pref[i+1] / self.p0) ** (R_D / C_PD)
                rho_avg = pref[i+1] / (R_D * T_avg)
                pref[i] = pref[i+1] + rho_avg * G * (-dz)  # +dz because dz is negative
        else:
            pref[0] = self.p_surface
        
        # Integrate upward from surface
        start_idx = max(first_above_surface, 0)
        if start_idx == 0:
            pref[0] = self.p_surface
        
        for i in range(start_idx + 1, len(zn)):
            dz = zn[i] - zn[i-1]
            theta_avg = 0.5 * (self._thref_on_grid[i] + self._thref_on_grid[i-1])
            T_avg = theta_avg * (pref[i-1] / self.p0) ** (R_D / C_PD)
            rho_avg = pref[i-1] / (R_D * T_avg)
            pref[i] = pref[i-1] - rho_avg * G * dz
        
        self._pref_on_grid = pref
        
        print(f"  Reference profiles computed:")
        print(f"    thref range: [{self._thref_on_grid.min():.1f}, {self._thref_on_grid.max():.1f}] K")
        print(f"    pref range: [{self._pref_on_grid.min():.0f}, {self._pref_on_grid.max():.0f}] Pa")
    
    def load_timestep(self, timestep: int) -> Dict[str, np.ndarray]:
        """
        Load all fields for a single timestep from MONC output.
        
        Performs all necessary conversions:
        1. Add reference profiles to perturbation fields
        2. Convert theta to theta_l
        3. Compute total water including ice species
        4. Transpose from (x, y, z) to (z, y, x)
        5. Interpolate w from z-grid to zn-grid
        
        Parameters
        ----------
        timestep : int
            Timestep index (0-based)
            
        Returns
        -------
        dict
            Dictionary with all required fields in CloudTracker format
        """
        if timestep < 0 or timestep >= len(self._files):
            raise IndexError(f"Timestep {timestep} out of range [0, {len(self._files)})")
        
        file_path = self._files[timestep]
        sim_time = self._times[timestep]
        print(f"Loading MONC data: {file_path.name} (t={sim_time}s)")
        
        # Ensure grid info is loaded
        grid = self.get_grid_info()
        xt, yt, zt = grid['xt'], grid['yt'], grid['zt']
        z_w = grid['z_w']
        
        with Dataset(file_path, 'r') as ds:
            # Find time dimension name (varies by timestep in MONC output)
            time_dims = [d for d in ds.dimensions if 'time' in d.lower()]
            if not time_dims:
                raise ValueError(f"No time dimension found in {file_path}")
            
            # Load raw fields
            # MONC dimension order: (time, x, y, z/zn)
            
            # Perturbation theta (on zn levels)
            th_pert = np.asarray(ds.variables['th'][0, :, :, :])
            
            # Pressure perturbation (on zn levels)
            # Note: We'll use reference pressure instead since p_pert is small
            p_pert = np.asarray(ds.variables['p'][0, :, :, :])
            
            # Velocities
            u = np.asarray(ds.variables['u'][0, :, :, :])  # on zn
            v = np.asarray(ds.variables['v'][0, :, :, :])  # on zn
            w_on_z = np.asarray(ds.variables['w'][0, :, :, :])  # on z (staggered)
            
            # Water species (all on zn levels, kg/kg)
            q_vapour = np.asarray(ds.variables['q_vapour'][0, :, :, :])
            q_cloud_liquid = np.asarray(ds.variables['q_cloud_liquid_mass'][0, :, :, :])
            q_rain = np.asarray(ds.variables['q_rain_mass'][0, :, :, :])
            
            # Ice species (for mixed-phase, may be zero for warm cases)
            q_ice = np.asarray(ds.variables['q_ice_mass'][0, :, :, :])
            q_snow = np.asarray(ds.variables['q_snow_mass'][0, :, :, :])
            q_graupel = np.asarray(ds.variables['q_graupel_mass'][0, :, :, :])
        
        # === CONVERSIONS ===
        
        # 1. Reconstruct full theta from perturbation
        # theta(x,y,z) = th_pert(x,y,z) + thref(z)
        # thref is 1D, broadcast over x,y
        theta = th_pert + self._thref_on_grid[np.newaxis, np.newaxis, :]
        
        # 2. Use reference pressure (perturbation is small for anelastic)
        # For more accuracy, could add p_pert, but it's typically << pref
        p = np.broadcast_to(
            self._pref_on_grid[np.newaxis, np.newaxis, :],
            theta.shape
        ).copy()  # Make writeable copy
        
        # 3. Convert theta to theta_l (liquid water potential temperature)
        # theta_l = theta - (L_v / c_pd) * (q_l / Pi)
        # where Pi = (p/p0)^(R_d/c_pd) is the Exner function
        Pi = (p / self.p0) ** (R_D / C_PD)
        theta_l = theta - (L_V / C_PD) * (q_cloud_liquid / Pi)
        
        # 4. Compute total water (including ice for mixed-phase)
        # q_t = q_vapour + q_cloud + q_ice + q_snow + q_graupel
        # Note: rain is separate for loading calculations
        q_t = q_vapour + q_cloud_liquid + q_ice + q_snow + q_graupel
        
        # 5. Cloud liquid water for cloud identification
        l = q_cloud_liquid
        
        # 6. Rain for loading calculations
        r = q_rain
        
        # 7. Interpolate w from z-grid to zn-grid (scalar levels)
        # Simple averaging of adjacent levels
        # w_on_z has shape (nx, ny, nz_w) where nz_w = len(z_w)
        # We need w on zn levels
        w = self._interpolate_w_to_scalar_levels(w_on_z, z_w, zt)
        
        # 8. Transpose from MONC (x, y, z) to CloudTracker (z, y, x)
        l = np.transpose(l, (2, 1, 0))
        u = np.transpose(u, (2, 1, 0))
        v = np.transpose(v, (2, 1, 0))
        w = np.transpose(w, (2, 1, 0))
        p = np.transpose(p, (2, 1, 0))
        theta_l = np.transpose(theta_l, (2, 1, 0))
        q_t = np.transpose(q_t, (2, 1, 0))
        r = np.transpose(r, (2, 1, 0))
        
        data = {
            'l': l,
            'u': u,
            'v': v,
            'w': w,
            'p': p,
            'theta_l': theta_l,
            'q_t': q_t,
            'r': r,
            'xt': xt,
            'yt': yt,
            'zt': zt,
        }
        
        # Validate the converted data
        try:
            self.validate_data(data)
        except ValueError as e:
            warnings.warn(f"Data validation warning: {e}")
        
        gc.collect()
        
        return data
    
    def _interpolate_w_to_scalar_levels(
        self, 
        w_on_z: np.ndarray, 
        z_w: np.ndarray, 
        zn: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate vertical velocity from w-levels (z) to scalar levels (zn).
        
        MONC uses a staggered Arakawa C-grid where w is on z-levels
        and scalars are on zn-levels (offset by ~half a grid cell).
        
        Parameters
        ----------
        w_on_z : np.ndarray
            Vertical velocity on z-levels, shape (nx, ny, nz)
        z_w : np.ndarray
            Heights of w-levels
        zn : np.ndarray
            Heights of scalar levels
            
        Returns
        -------
        np.ndarray
            Vertical velocity interpolated to scalar levels, shape (nx, ny, nz)
        """
        nx, ny, nz_w = w_on_z.shape
        nz = len(zn)
        
        # Simple linear interpolation for each column
        w_on_zn = np.zeros((nx, ny, nz), dtype=w_on_z.dtype)
        
        for k, z_target in enumerate(zn):
            # Find bracketing levels in z_w
            if z_target <= z_w[0]:
                # Below first w-level, use first w value
                w_on_zn[:, :, k] = w_on_z[:, :, 0]
            elif z_target >= z_w[-1]:
                # Above last w-level, use last w value
                w_on_zn[:, :, k] = w_on_z[:, :, -1]
            else:
                # Linear interpolation
                idx = np.searchsorted(z_w, z_target) - 1
                idx = max(0, min(idx, nz_w - 2))
                
                z_below = z_w[idx]
                z_above = z_w[idx + 1]
                weight = (z_target - z_below) / (z_above - z_below)
                
                w_on_zn[:, :, k] = (1 - weight) * w_on_z[:, :, idx] + weight * w_on_z[:, :, idx + 1]
        
        return w_on_zn
    
    def load_mean_velocities(self, timestep: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load horizontally-averaged wind profiles efficiently.
        
        Only loads u and v fields instead of all fields.
        """
        if timestep < 0 or timestep >= len(self._files):
            raise IndexError(f"Timestep {timestep} out of range")
        
        file_path = self._files[timestep]
        
        with Dataset(file_path, 'r') as ds:
            u = np.asarray(ds.variables['u'][0, :, :, :])
            v = np.asarray(ds.variables['v'][0, :, :, :])
        
        # MONC order is (x, y, z), average over x and y
        mean_u = np.mean(u, axis=(0, 1))
        mean_v = np.mean(v, axis=(0, 1))
        
        del u, v
        gc.collect()
        
        return mean_u, mean_v
    
    def get_reference_profiles(self) -> Dict[str, np.ndarray]:
        """
        Get the reference profiles used for reconstruction.
        
        Returns
        -------
        dict
            Dictionary with 'z', 'thref', 'pref' arrays
        """
        grid = self.get_grid_info()
        return {
            'z': grid['zt'],
            'thref': self._thref_on_grid,
            'pref': self._pref_on_grid,
        }
