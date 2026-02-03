"""
Parser for MONC configuration files (.mcf format).

The .mcf format is a simple key=value format used by MONC LES model:
- key=value for simple values
- key=val1,val2,val3 for comma-separated arrays
- key="string value" for quoted strings
- # for comments
- Fortran-style booleans: .true., .false.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np


class MONCConfigParser:
    """
    Parser for MONC .mcf configuration files.
    
    Extracts parameters needed for CloudTracker MONC adapter:
    - Grid configuration (dxx, dyy, x_size, y_size, z_size)
    - Reference pressure (surface_pressure, surface_reference_pressure)
    - Reference theta profile (z_init_pl_theta, f_init_pl_theta)
    
    Example usage:
        parser = MONCConfigParser('/path/to/config.mcf')
        config = parser.parse()
        
        dx = config['dxx']
        thref_z = config['z_init_pl_theta']
        thref_f = config['f_init_pl_theta']
    """
    
    # Required parameters for CloudTracker
    REQUIRED_PARAMS = [
        'dxx', 'dyy',
        'surface_pressure', 'surface_reference_pressure',
        'z_init_pl_theta', 'f_init_pl_theta',
    ]
    
    # Parameters that should be parsed as arrays
    ARRAY_PARAMS = [
        'z_init_pl_theta', 'f_init_pl_theta',
        'z_init_pl_u', 'f_init_pl_u',
        'z_init_pl_v', 'f_init_pl_v',
        'z_init_pl_q', 'f_init_pl_q',
        'kgd', 'hgd',
        'surface_boundary_input_times',
        'surface_latent_heat_flux',
        'surface_sensible_heat_flux',
    ]
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the parser with a path to the .mcf file.
        
        Parameters
        ----------
        config_path : str or Path
            Path to the MONC configuration file (.mcf)
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"MONC config file not found: {self.config_path}")
        
        self._raw_config: Dict[str, str] = {}
        self._parsed_config: Dict[str, Any] = {}
    
    def parse(self) -> Dict[str, Any]:
        """
        Parse the .mcf file and return a dictionary of configuration values.
        
        Returns
        -------
        dict
            Parsed configuration with appropriate Python types
            
        Raises
        ------
        ValueError
            If required parameters are missing
        """
        self._read_raw_config()
        self._convert_types()
        self._validate_required_params()
        return self._parsed_config
    
    def _read_raw_config(self) -> None:
        """Read the .mcf file and extract key=value pairs."""
        with open(self.config_path, 'r') as f:
            content = f.read()
        
        # Remove comments (lines starting with # or inline comments)
        lines = []
        for line in content.split('\n'):
            # Remove inline comments (but preserve # in strings)
            if '#' in line and not ('"' in line or "'" in line):
                line = line.split('#')[0]
            elif '#' in line:
                # Handle case where # might be in a string - simple heuristic
                # Find first # that's not inside quotes
                in_quote = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in '"\'':
                        if not in_quote:
                            in_quote = True
                            quote_char = char
                        elif char == quote_char:
                            in_quote = False
                    elif char == '#' and not in_quote:
                        line = line[:i]
                        break
            lines.append(line.strip())
        
        # Parse key=value pairs
        for line in lines:
            if '=' in line and not line.startswith('#'):
                # Split on first = only
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Skip empty keys
                if key:
                    self._raw_config[key] = value
    
    def _convert_types(self) -> None:
        """Convert raw string values to appropriate Python types."""
        for key, value in self._raw_config.items():
            self._parsed_config[key] = self._parse_value(key, value)
    
    def _parse_value(self, key: str, value: str) -> Any:
        """
        Parse a single value string into the appropriate Python type.
        
        Parameters
        ----------
        key : str
            Parameter name (used to determine if it should be an array)
        value : str
            Raw string value from config file
            
        Returns
        -------
        Any
            Parsed value (bool, int, float, str, or numpy array)
        """
        # Handle empty values
        if not value:
            return None
        
        # Handle quoted strings
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        
        # Handle Fortran booleans
        if value.lower() in ['.true.', '.t.', 'true']:
            return True
        if value.lower() in ['.false.', '.f.', 'false']:
            return False
        
        # Handle arrays (comma-separated values)
        if ',' in value or key in self.ARRAY_PARAMS:
            return self._parse_array(value)
        
        # Handle single numeric values
        return self._parse_numeric(value)
    
    def _parse_array(self, value: str) -> np.ndarray:
        """Parse a comma-separated string into a numpy array."""
        # Split by comma and strip whitespace
        parts = [p.strip() for p in value.split(',') if p.strip()]
        
        # Convert each part to numeric
        numeric_values = []
        for part in parts:
            numeric_values.append(self._parse_numeric(part))
        
        return np.array(numeric_values)
    
    def _parse_numeric(self, value: str) -> Union[int, float, str]:
        """Parse a string into int, float, or return as string if not numeric."""
        # Remove trailing dots (e.g., "0.0." -> "0.0")
        value = value.rstrip('.')
        
        # Try integer first
        try:
            # Check if it looks like an integer (no decimal point or exponent)
            if '.' not in value and 'e' not in value.lower():
                return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string if not numeric
        return value
    
    def _validate_required_params(self) -> None:
        """Check that all required parameters are present."""
        missing = []
        for param in self.REQUIRED_PARAMS:
            if param not in self._parsed_config:
                missing.append(param)
        
        if missing:
            raise ValueError(
                f"Missing required parameters in MONC config file: {missing}\n"
                f"File: {self.config_path}"
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value with optional default.
        
        Parameters
        ----------
        key : str
            Parameter name
        default : Any, optional
            Default value if parameter not found
            
        Returns
        -------
        Any
            Parameter value or default
        """
        return self._parsed_config.get(key, default)
    
    def get_reference_theta_profile(self) -> tuple:
        """
        Get the reference theta profile arrays.
        
        Returns
        -------
        tuple
            (z_array, theta_array) as numpy arrays in meters and Kelvin
        """
        z = self._parsed_config.get('z_init_pl_theta')
        f = self._parsed_config.get('f_init_pl_theta')
        
        if z is None or f is None:
            raise ValueError("Reference theta profile not found in config")
        
        return np.asarray(z), np.asarray(f)
    
    def get_grid_spacing(self) -> tuple:
        """
        Get horizontal grid spacing.
        
        Returns
        -------
        tuple
            (dx, dy) in meters
        """
        dx = self._parsed_config.get('dxx')
        dy = self._parsed_config.get('dyy')
        
        if dx is None or dy is None:
            raise ValueError("Grid spacing (dxx, dyy) not found in config")
        
        return float(dx), float(dy)
    
    def get_surface_pressures(self) -> tuple:
        """
        Get surface pressure values.
        
        Returns
        -------
        tuple
            (surface_pressure, reference_pressure) in Pa
        """
        p_surf = self._parsed_config.get('surface_pressure')
        p_ref = self._parsed_config.get('surface_reference_pressure')
        
        if p_surf is None or p_ref is None:
            raise ValueError("Surface pressures not found in config")
        
        return float(p_surf), float(p_ref)
    
    def get_grid_dimensions(self) -> tuple:
        """
        Get grid dimensions.
        
        Returns
        -------
        tuple
            (x_size, y_size, z_size)
        """
        x = self._parsed_config.get('x_size')
        y = self._parsed_config.get('y_size')
        z = self._parsed_config.get('z_size')
        
        return int(x) if x else None, int(y) if y else None, int(z) if z else None
    
    def __repr__(self) -> str:
        return f"MONCConfigParser('{self.config_path}')"
