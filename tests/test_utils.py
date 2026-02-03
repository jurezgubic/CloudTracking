"""
Test utilities for cloud tracking tests.

Provides a MockCloudField class that properly builds the KD-tree infrastructure
required by CloudTracker.is_match() and update_tracks().
"""

import numpy as np
from scipy.spatial import cKDTree
from lib.cloud import Cloud


class MockCloudField:
    """
    Mock CloudField that builds the KD-tree infrastructure for testing.
    
    This class mimics the essential behavior of CloudField for testing purposes,
    including building the surface point KD-tree required by CloudTracker.is_match().
    """
    
    def __init__(self, clouds_dict, timestep=0):
        """
        Initialize MockCloudField with a dictionary of clouds.
        
        Parameters:
        -----------
        clouds_dict : dict
            Dictionary mapping cloud_id to Cloud objects
        timestep : int
            Timestep for this cloud field
        """
        self.clouds = clouds_dict
        self.timestep = timestep
        
        # Build KD-tree infrastructure (mimics CloudField.build_global_surface_kdtree)
        self.surface_points_array = None
        self.surface_point_to_cloud_id = None
        self.surface_points_kdtree = None
        self._build_surface_kdtree()
    
    def _build_surface_kdtree(self):
        """Build the KD-tree from cloud surface points."""
        if not self.clouds:
            return
        
        # Calculate total number of surface points
        total_points = 0
        for cloud in self.clouds.values():
            if hasattr(cloud, 'surface_points') and cloud.surface_points is not None:
                if isinstance(cloud.surface_points, np.ndarray):
                    total_points += cloud.surface_points.shape[0]
                else:
                    total_points += len(cloud.surface_points)
        
        if total_points == 0:
            return
        
        # Pre-allocate arrays
        self.surface_points_array = np.empty((total_points, 3), dtype=np.float32)
        self.surface_point_to_cloud_id = np.empty(total_points, dtype=object)
        
        # Fill arrays with surface points and their cloud IDs
        idx = 0
        for cloud_id, cloud in self.clouds.items():
            if hasattr(cloud, 'surface_points') and cloud.surface_points is not None:
                points = cloud.surface_points
                if not isinstance(points, np.ndarray):
                    points = np.array(points)
                n_points = points.shape[0] if points.ndim > 1 else 1
                if points.ndim == 1:
                    points = points.reshape(1, -1)
                if n_points > 0:
                    self.surface_points_array[idx:idx+n_points] = points
                    self.surface_point_to_cloud_id[idx:idx+n_points] = cloud_id
                    idx += n_points
        
        # Trim arrays to actual size
        if idx < total_points:
            self.surface_points_array = self.surface_points_array[:idx]
            self.surface_point_to_cloud_id = self.surface_point_to_cloud_id[:idx]
        
        # Build the KD-tree using only X,Y coordinates
        if len(self.surface_points_array) > 0:
            self.surface_points_kdtree = cKDTree(self.surface_points_array[:, :2])


def create_test_cloud(cloud_id, x, y, z, size=10, is_active=True, age=0,
                      mean_u=None, mean_v=None, mean_w=None, zt=None):
    """
    Create a Cloud object for testing with minimal required attributes.
    
    Parameters:
    -----------
    cloud_id : int
        Unique identifier for the cloud
    x, y, z : float
        Location coordinates
    size : int
        Cloud size (number of points)
    is_active : bool
        Whether the cloud is active
    age : int
        Cloud age in timesteps
    mean_u, mean_v, mean_w : array-like or None
        Mean velocity profiles. If None, uses zeros.
    zt : array-like or None
        Height levels. If None, uses default.
        
    Returns:
    --------
    Cloud : Cloud object suitable for testing
    """
    if zt is None:
        zt = np.linspace(0, 1000, 10)
    if mean_u is None:
        mean_u = np.zeros_like(zt)
    if mean_v is None:
        mean_v = np.zeros_like(zt)
    if mean_w is None:
        mean_w = np.zeros_like(zt)
    
    # Create surface points as numpy array
    # Add some spread around the center point for more realistic surface
    n_surface_points = max(5, size // 2)
    surface_points = np.array([
        [x + dx, y + dy, z]
        for dx in range(-2, 3)
        for dy in range(-2, 3)
    ][:n_surface_points], dtype=np.float32)
    
    points = [(x, y, z)]
    
    return Cloud(
        cloud_id=cloud_id,
        size=size,
        surface_area=size * 2,
        cloud_base_area=size // 2,
        cloud_base_height=z,
        location=(x, y, z),
        points=points,
        surface_points=surface_points,
        timestep=0,
        max_height=z,
        max_w=1.0,
        max_w_cloud_base=0.5,
        mean_u=mean_u,
        mean_v=mean_v,
        mean_w=mean_w,
        ql_flux=0.1,
        mass_flux=0.2,
        mass_flux_per_level=np.zeros_like(zt),
        temp_per_level=np.zeros_like(zt),
        theta_outside_per_level=np.zeros_like(zt),
        w_per_level=np.zeros_like(zt),
        circum_per_level=np.zeros_like(zt),
        eff_radius_per_level=np.zeros_like(zt),
        is_active=is_active,
        age=age
    )


def get_test_config(use_pre_filtering=False, **kwargs):
    """
    Get a test configuration dictionary with sensible defaults.
    
    Parameters:
    -----------
    use_pre_filtering : bool
        Whether to use pre-filtering in tracking
    **kwargs : dict
        Additional config values to override defaults
        
    Returns:
    --------
    dict : Configuration dictionary for CloudTracker
    """
    config = {
        'min_size': 3,
        'timestep_duration': 60,
        'horizontal_resolution': 25.0,
        'switch_wind_drift': False,
        'switch_background_drift': False,
        'switch_vertical_drift': False,
        'max_expected_cloud_speed': 20.0,
        'bounding_box_safety_factor': 2.0,
        'use_pre_filtering': use_pre_filtering,
        'switch_prefilter_fallback': True,
        'match_safety_factor_dynamic': 2.0,
        'min_h_match_factor': 1.0,
        'min_v_match_factor': 1.0,
        'min_surface_overlap_points': 1,
    }
    config.update(kwargs)
    return config
