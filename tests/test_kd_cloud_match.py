"""
Tests for KD-tree based cloud matching.

These tests verify that CloudTracker.is_match() correctly identifies
cloud matches between timesteps using surface point overlap.
"""
import unittest
import numpy as np
from lib.cloud import Cloud
from lib.cloudtracker import CloudTracker
from tests.test_utils import MockCloudField


class TestCloudMatching(unittest.TestCase):
    """Test cloud matching using KD-tree based surface point overlap."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Define config with known values
        self.config = {
            'horizontal_resolution': 25.0,
            'timestep_duration': 60,
            'switch_background_drift': False,
            'switch_wind_drift': False,
            'switch_vertical_drift': False,
            'max_expected_cloud_speed': 20.0,
            'bounding_box_safety_factor': 2.0,
            'use_pre_filtering': False,
            'match_safety_factor_dynamic': 2.0,
            'min_h_match_factor': 1.0,
            'min_v_match_factor': 1.0,
            'min_surface_overlap_points': 1,
        }
        
        # Create CloudTracker instance and set up domain info
        self.cloud_tracker = CloudTracker(self.config)
        self.cloud_tracker.zt = np.array([100, 300, 500, 700, 900])
        self.cloud_tracker.xt = np.linspace(0, 1000, 41)  # 25m resolution
        self.cloud_tracker.yt = np.linspace(0, 1000, 41)
        self.cloud_tracker.domain_size_x = 1000.0
        self.cloud_tracker.domain_size_y = 1000.0
        
    def _create_cloud(self, cloud_id, location, surface_points, 
                      mean_u=0.0, mean_v=0.0, mean_w=0.0, is_active=True):
        """
        Helper to create Cloud with all required parameters.
        
        Parameters:
        -----------
        cloud_id : int
            Unique identifier
        location : tuple
            (x, y, z) center location
        surface_points : list or np.ndarray
            Surface point coordinates as list of (x, y, z) tuples
        mean_u, mean_v, mean_w : float
            Mean velocities (scalar values used for drift prediction)
        is_active : bool
            Whether cloud is active
        """
        if not isinstance(surface_points, np.ndarray):
            surface_points = np.array(surface_points, dtype=np.float32)
        
        return Cloud(
            cloud_id=cloud_id,
            size=100,
            surface_area=50,
            cloud_base_area=20,
            cloud_base_height=location[2],
            location=location,
            points=[location],
            surface_points=surface_points,
            timestep=0,
            max_height=location[2],
            max_w=1.0,
            max_w_cloud_base=0.5,
            mean_u=mean_u,  # Scalar: cloud's mean u-velocity
            mean_v=mean_v,  # Scalar: cloud's mean v-velocity
            mean_w=mean_w,  # Scalar: cloud's mean w-velocity
            ql_flux=0.1,
            mass_flux=0.2,
            mass_flux_per_level=[0.2],
            temp_per_level=[290],
            theta_outside_per_level=[300],
            w_per_level=[1.0],
            circum_per_level=[50],
            eff_radius_per_level=[20],
            is_active=is_active
        )
        
    def test_exact_match(self):
        """Test when clouds have overlapping surface points."""
        # Cloud at t0 with surface points around (100, 100, 500)
        surface_pts_t0 = [(100, 100, 500), (125, 100, 500), (100, 125, 500)]
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            surface_points=surface_pts_t0,
        )
        
        # Cloud at t1 with overlapping surface points
        # Same location = should match
        surface_pts_t1 = [(100, 100, 500), (125, 100, 500), (100, 125, 500)]
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(100, 100, 500),
            surface_points=surface_pts_t1,
        )
        
        # Build mock cloud field with cloud_t1 (current timestep)
        mock_cloud_field = MockCloudField({cloud_t1.cloud_id: cloud_t1})
        
        self.assertTrue(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                       "Should match when cloud surface points overlap exactly")

    def test_nearby_match(self):
        """Test when clouds are slightly offset but within threshold."""
        # Cloud at t0
        surface_pts_t0 = [(100, 100, 500)]
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            surface_points=surface_pts_t0,
        )
        
        # Cloud at t1 - nearby but within horizontal threshold (25m)
        # Distance = √(10² + 10²) ≈ 14.14, which is < 25
        surface_pts_t1 = [(110, 110, 500)]
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(110, 110, 500),
            surface_points=surface_pts_t1,
        )
        
        mock_cloud_field = MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.assertTrue(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                       "Should match when cloud points are within threshold distance")

    def test_no_match_distance(self):
        """Test when clouds are too far apart to match."""
        # Cloud at t0
        surface_pts_t0 = [(100, 100, 500)]
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            surface_points=surface_pts_t0,
        )
        
        # Cloud at t1 - too far away (beyond threshold of 25m)
        # Distance = 50m which is > 25m
        surface_pts_t1 = [(150, 100, 500)]
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(150, 100, 500),
            surface_points=surface_pts_t1,
        )
        
        mock_cloud_field = MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.assertFalse(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                        "Should not match when cloud points are beyond threshold distance")
    
    def test_drift_prediction_match(self):
        """Test that cloud velocity is used to predict position."""
        # Cloud at t0 moving at 0.4 m/s in x direction (24m in 60s)
        # The predicted location after 60s will be x=124
        surface_pts_t0 = [(100, 100, 500)]
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            surface_points=surface_pts_t0,
            mean_u=0.4,  # 0.4 m/s * 60s = 24m displacement
            mean_v=0.0,
            mean_w=0.0,
        )
        
        # Cloud at t1 is at x=124 (where we predicted t0 would be)
        # This should match because drift-adjusted t0 points overlap with t1
        surface_pts_t1 = [(124, 100, 500)]
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(124, 100, 500),
            surface_points=surface_pts_t1,
        )
        
        mock_cloud_field = MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.assertTrue(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                       "Should match when drift-predicted location overlaps with new cloud")
                       
    def test_vertical_drift_match(self):
        """Test that vertical drift is correctly applied for matching."""
        # Cloud at t0 with upward velocity
        surface_pts_t0 = [(100, 100, 500)]
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            surface_points=surface_pts_t0,
            mean_u=0.0,
            mean_v=0.0,
            mean_w=0.3,  # 0.3 m/s * 60s = 18m vertical movement
        )
        
        # Cloud at t1 - at the predicted vertical location
        surface_pts_t1 = [(100, 100, 518)]  # 500 + 18
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(100, 100, 518),
            surface_points=surface_pts_t1,
        )
        
        mock_cloud_field = MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.assertTrue(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                       "Should match when cloud points move with vertical drift")
                       
    def test_large_vertical_movement_no_match(self):
        """Test that clouds with too much vertical movement don't match."""
        # Cloud at t0 (stationary - no predicted vertical movement)
        surface_pts_t0 = [(100, 100, 500)]
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            surface_points=surface_pts_t0,
            mean_u=0.0,
            mean_v=0.0,
            mean_w=0.0,  # No vertical velocity
        )
        
        # Cloud at t1 - with large unexpected vertical movement
        # Vertical threshold is at least 25m (min_v_match_factor * horizontal_resolution)
        # 50m exceeds this threshold
        surface_pts_t1 = [(100, 100, 550)]  # 50m vertical offset
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(100, 100, 550),
            surface_points=surface_pts_t1,
        )
        
        mock_cloud_field = MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.assertFalse(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                        "Should not match when vertical movement exceeds threshold")

    def test_inactive_cloud_no_match(self):
        """Test that inactive clouds are not matched."""
        surface_pts_t0 = [(100, 100, 500)]
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            surface_points=surface_pts_t0,
            is_active=False,  # Inactive
        )
        
        surface_pts_t1 = [(100, 100, 500)]
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(100, 100, 500),
            surface_points=surface_pts_t1,
        )
        
        mock_cloud_field = MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.assertFalse(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                        "Should not match when previous cloud is inactive")

    def test_empty_surface_points_no_match(self):
        """Test that clouds with empty surface points don't match."""
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            surface_points=np.array([], dtype=np.float32).reshape(0, 3),  # Empty
        )
        
        surface_pts_t1 = [(100, 100, 500)]
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(100, 100, 500),
            surface_points=surface_pts_t1,
        )
        
        mock_cloud_field = MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.assertFalse(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                        "Should not match when previous cloud has no surface points")


if __name__ == '__main__':
    unittest.main()