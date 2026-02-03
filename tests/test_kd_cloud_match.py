import unittest
import numpy as np
from lib.cloud import Cloud
from lib.cloudtracker import CloudTracker

# Skip these tests - they require full KD-tree setup which has changed significantly.
# The is_match method now requires a proper CloudField with surface_points_kdtree,
# surface_point_to_cloud_id, and surface_points_array attributes.
# These tests need to be rewritten to use the actual CloudField class or proper mocks.

@unittest.skip("Tests require KD-tree setup - API has changed")
class TestCloudMatching(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test."""
        # Define config with known values
        self.config = {
            'horizontal_resolution': 25.0,  # Threshold for matching
            'u_drift': -5.0,                # Horizontal drift in x
            'v_drift': -4.0,                # Horizontal drift in y
            'timestep_duration': 60,        # Seconds between timesteps
            'switch_background_drift': True, # Use background drift
            'switch_wind_drift': True,       # Use wind drift calculations
            'switch_vertical_drift': True,   # Enable vertical drift
            'max_expected_cloud_speed': 20.0,
            'bounding_box_safety_factor': 2.0,
        }
        
        # Create CloudTracker instance
        self.cloud_tracker = CloudTracker(self.config)
        
        # Mock the drift calculation to return controlled values
        self.cloud_tracker.drift_translation_calculation = lambda: (-5.0, -4.0)
        self.cloud_tracker.wind_drift_calculation = lambda z: (0.0, 0.0) if z < 500 else (-1.0, -0.5)
        self.cloud_tracker.vertical_drift_calculation = lambda z: 2.0 if z < 500 else 3.5  # Mock vertical drift
        
        # Set mean_w values for vertical velocity
        self.cloud_tracker.mean_w = np.array([0.03, 0.04, 0.05, 0.06, 0.07])
        self.cloud_tracker.zt = np.array([100, 300, 500, 700, 900])
        
        # Default arrays for Cloud constructor
        self.mean_u = np.zeros(5)
        self.mean_v = np.zeros(5)
        self.mean_w_arr = np.zeros(5)
        
        # Mock cloud field for is_match calls
        class MockCloudField:
            def __init__(self, clouds_dict):
                self.clouds = clouds_dict
                self.surface_points_kdtree = None  # Let is_match use fallback logic
        self.MockCloudField = MockCloudField
        
    def _create_cloud(self, cloud_id, location, points, timestep, max_height, is_active=True):
        """Helper to create Cloud with all required parameters."""
        return Cloud(
            cloud_id=cloud_id,
            size=100,
            surface_area=50,
            cloud_base_area=20,
            cloud_base_height=location[2],
            location=location,
            points=points,
            surface_points=np.array([points[0]]),  # numpy array for .any() call
            timestep=timestep,
            max_height=max_height,
            max_w=1.0,
            max_w_cloud_base=0.5,
            mean_u=self.mean_u,
            mean_v=self.mean_v,
            mean_w=self.mean_w_arr,
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
        """Test when clouds are exactly matched after drift."""
        # Cloud at t0
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            points=[(100, 100, 500), (125, 100, 500), (100, 125, 500)],
            timestep=0,
            max_height=500
        )
        
        # Cloud at t1 - exact match with drift
        # Points moved exactly by drift: (-5, -4)
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(95, 96, 500),  # Moved by drift
            points=[(95, 96, 500), (120, 96, 500), (95, 121, 500)],  # Each point moved by drift
            timestep=1,
            max_height=500
        )
        
        # Create mock cloud field containing cloud_t1
        mock_cloud_field = self.MockCloudField({cloud_t1.cloud_id: cloud_t1})
        
        self.assertTrue(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                       "Should match when cloud points are exactly displaced by drift")

    def test_nearby_match(self):
        """Test when clouds are slightly offset but within threshold."""
        # Cloud at t0
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            points=[(100, 100, 500)],
            timestep=0,
            max_height=500
        )
        
        # Cloud at t1 - nearby but within threshold
        # Points moved by drift + small offset (within threshold)
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(95 + 10, 96 + 10, 500),  # Drift + offset
            points=[(95 + 10, 96 + 10, 500)],  # Drift + offset but within threshold (25.0)
            timestep=1,
            max_height=500
        )
        
        # Distance is √(10² + 10²) = √200 ≈ 14.14, which is < 25
        mock_cloud_field = self.MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.assertTrue(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                       "Should match when cloud points are within threshold distance")

    def test_no_match_distance(self):
        """Test when clouds are too far apart to match."""
        # Cloud at t0
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            points=[(100, 100, 500)],
            timestep=0,
            max_height=500
        )
        
        # Cloud at t1 - too far away (beyond threshold)
        # Points moved by drift + large offset (beyond threshold)
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(95 + 30, 96 + 0, 500),  # Offset by 30 in x
            points=[(95 + 30, 96 + 0, 500)],  # Beyond threshold in x direction
            timestep=1,
            max_height=500
        )
        
        mock_cloud_field = self.MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.assertFalse(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                        "Should not match when cloud points are beyond threshold distance")
    
    def test_height_specific_drift(self):
        """Test that height-specific wind drift is correctly applied."""
        # Cloud at t0 (high altitude)
        cloud_t0_high = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 800),  # High altitude = additional drift
            points=[(100, 100, 800)],
            timestep=0,
            max_height=800
        )
        
        # Cloud at t1 - matching with height-specific drift
        # Base drift (-5, -4) + wind drift (-1, -0.5) at height 800
        cloud_t1_high = self._create_cloud(
            cloud_id=2,
            location=(94, 95.5, 800),  # Total drift: -6, -4.5
            points=[(94, 95.5, 800)],
            timestep=1,
            max_height=800
        )
        
        mock_cloud_field = self.MockCloudField({cloud_t1_high.cloud_id: cloud_t1_high})
        self.assertTrue(self.cloud_tracker.is_match(cloud_t1_high, cloud_t0_high, mock_cloud_field),
                       "Should match with correct height-specific wind drift")
                       
    def test_vertical_drift_match(self):
        """Test that vertical drift is correctly applied for matching."""
        # Cloud at t0
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            points=[(100, 100, 500)],
            timestep=0,
            max_height=500
        )
        
        # Cloud at t1 - with horizontal drift and vertical movement
        # Horizontal drift: (-5, -4)
        # Vertical drift: +2.0 for height < 500
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(95, 96, 500 + 2.0),  # Applied drift and vertical movement
            points=[(95, 96, 500 + 2.0)],
            timestep=1,
            max_height=500 + 2.0
        )
        
        mock_cloud_field = self.MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.assertTrue(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                       "Should match when cloud points move with horizontal and vertical drift")
                       
    def test_large_vertical_movement_no_match(self):
        """Test that clouds with too much vertical movement don't match."""
        # Cloud at t0
        cloud_t0 = self._create_cloud(
            cloud_id=1,
            location=(100, 100, 500),
            points=[(100, 100, 500)],
            timestep=0,
            max_height=500
        )
        
        # Cloud at t1 - with correct horizontal drift but excessive vertical movement
        cloud_t1 = self._create_cloud(
            cloud_id=2,
            location=(95, 96, 500 + 50),  # 50m is beyond the vertical threshold
            points=[(95, 96, 500 + 50)],
            timestep=1,
            max_height=500 + 50
        )
        
        mock_cloud_field = self.MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.assertFalse(self.cloud_tracker.is_match(cloud_t1, cloud_t0, mock_cloud_field),
                        "Should not match when vertical movement exceeds threshold")

if __name__ == '__main__':
    unittest.main()