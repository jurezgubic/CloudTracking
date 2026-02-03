import unittest
import numpy as np
import sys
import os

# This test verifies three key behaviors:
# 1. Track Continuation: One fragment from a split cloud continues the original track and has its age incremented
# 2. Split Inheritance: Other fragments start new tracks but inherit the parent cloud's age + 1
# 3. New Cloud Handling: Genuinely new clouds (without a parent) start with age 0
# The test mocks the spatial matching by overriding is_match to have precise control over which clouds are considered matches. 

# Add parent directory to path to import from lib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.cloudtracker import CloudTracker
from lib.cloud import Cloud

class MockCloudField:
    """Simple mock of CloudField for testing"""
    def __init__(self, clouds_dict, timestep=0):
        self.clouds = clouds_dict
        self.timestep = timestep

class TestCloudSplitting(unittest.TestCase):
    """Test the cloud splitting behavior in CloudTracker"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            'min_size': 3,
            'timestep_duration': 60,
            'horizontal_resolution': 25.0,
            'switch_wind_drift': False,
            'switch_background_drift': False,
            'switch_vertical_drift': False,
            'max_expected_cloud_speed': 20.0,
            'bounding_box_safety_factor': 2.0,
            'use_pre_filtering': False,  # Disable pre-filtering for simpler tests
        }
        
    def test_cloud_splitting(self):
        """Test how CloudTracker handles cloud splitting"""
        # Initialize tracker
        tracker = CloudTracker(self.config)
        
        # Create mock height levels
        zt = np.array([0, 100, 200, 300, 400, 500])
        xt = np.arange(0, 500, 25.0)  # x coordinates
        yt = np.arange(0, 500, 25.0)  # y coordinates
        mean_u = np.zeros_like(zt)
        mean_v = np.zeros_like(zt)
        mean_w = np.zeros_like(zt)
        
        # TIMESTEP 1: Create one cloud
        cloud1 = Cloud(
            cloud_id=1,
            size=10,
            surface_area=20,
            cloud_base_area=5,
            cloud_base_height=200,
            location=(100, 100, 200),
            points=[(100, 100, 200), (101, 100, 200), (100, 101, 200), (101, 101, 200)],
            surface_points=np.array([(100, 100, 200)]),
            timestep=0,
            max_height=200,
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
            eff_radius_per_level=np.zeros_like(zt)
        )
        
        # Add the cloud to a mock CloudField
        cloud_field1 = MockCloudField({1: cloud1})
        
        # Update tracker with the first cloud field
        tracker.update_tracks(cloud_field1, zt, xt, yt)
        
        # TIMESTEP 2: Create split clouds and a new cloud
        # First split fragment (close to original cloud)
        cloud2 = Cloud(
            cloud_id=2,
            size=6,
            surface_area=12,
            cloud_base_area=3,
            cloud_base_height=210,
            location=(102, 102, 210),
            points=[(102, 102, 210), (103, 102, 210)],
            surface_points=np.array([(102, 102, 210)]),
            timestep=1,
            max_height=210,
            max_w=1.1,
            max_w_cloud_base=0.6,
            mean_u=mean_u,
            mean_v=mean_v,
            mean_w=mean_w,
            ql_flux=0.11,
            mass_flux=0.21,
            mass_flux_per_level=np.zeros_like(zt),
            temp_per_level=np.zeros_like(zt),
            theta_outside_per_level=np.zeros_like(zt),
            w_per_level=np.zeros_like(zt),
            circum_per_level=np.zeros_like(zt),
            eff_radius_per_level=np.zeros_like(zt)
        )
        
        # Second split fragment (also close to original cloud)
        cloud3 = Cloud(
            cloud_id=3,
            size=4,
            surface_area=8,
            cloud_base_area=2,
            cloud_base_height=205,
            location=(97, 99, 205),
            points=[(97, 99, 205), (98, 99, 205)],
            surface_points=np.array([(97, 99, 205)]),
            timestep=1,
            max_height=205,
            max_w=0.9,
            max_w_cloud_base=0.4,
            mean_u=mean_u,
            mean_v=mean_v,
            mean_w=mean_w,
            ql_flux=0.09,
            mass_flux=0.19,
            mass_flux_per_level=np.zeros_like(zt),
            temp_per_level=np.zeros_like(zt),
            theta_outside_per_level=np.zeros_like(zt),
            w_per_level=np.zeros_like(zt),
            circum_per_level=np.zeros_like(zt),
            eff_radius_per_level=np.zeros_like(zt)
        )
        
        # Completely new cloud (far from original cloud)
        cloud4 = Cloud(
            cloud_id=4,
            size=8,
            surface_area=16,
            cloud_base_area=4,
            cloud_base_height=250,
            location=(300, 300, 250),
            points=[(300, 300, 250), (301, 300, 250)],
            surface_points=np.array([(300, 300, 250)]),
            timestep=1,
            max_height=250,
            max_w=1.2,
            max_w_cloud_base=0.7,
            mean_u=mean_u,
            mean_v=mean_v,
            mean_w=mean_w,
            ql_flux=0.12,
            mass_flux=0.22,
            mass_flux_per_level=np.zeros_like(zt),
            temp_per_level=np.zeros_like(zt),
            theta_outside_per_level=np.zeros_like(zt),
            w_per_level=np.zeros_like(zt),
            circum_per_level=np.zeros_like(zt),
            eff_radius_per_level=np.zeros_like(zt)
        )
        
        # Add the clouds to a mock CloudField
        cloud_field2 = MockCloudField({2: cloud2, 3: cloud3, 4: cloud4})
        
        # Override is_match method to control which clouds are considered matches
        # This simulates the spatial proximity matching without the complexity
        original_is_match = tracker.is_match
        
        def mock_is_match(cloud, last_cloud_in_track, current_cloud_field):
            # Cloud2 and Cloud3 should match with Cloud1
            # Cloud4 should not match with any previous cloud
            if last_cloud_in_track.cloud_id == 1 and cloud.cloud_id in [2, 3]:
                return True
            return False
        
        # Apply mock method
        tracker.is_match = mock_is_match
        
        # Update tracker with the second cloud field
        tracker.update_tracks(cloud_field2, zt, xt, yt)
        
        # Restore original is_match method
        tracker.is_match = original_is_match
        
        # Get all tracks
        tracks = tracker.get_tracks()
        
        # Check results
        # First, find which track contains cloud1
        original_track_id = None
        for track_id, track in tracks.items():
            if any(c.cloud_id == 1 for c in track):
                original_track_id = track_id
                break
        
        self.assertIsNotNone(original_track_id, "Could not find track containing the original cloud")
        
        # Check that one split fragment continues the original track
        # and has age = 1 (original age 0 + 1)
        original_track = tracks[original_track_id]
        self.assertEqual(len(original_track), 2, "Original track should have 2 clouds")
        self.assertEqual(original_track[0].cloud_id, 1, "First cloud in original track should be cloud1")
        self.assertTrue(original_track[1].cloud_id in [2, 3], "Second cloud should be one of the split fragments")
        self.assertEqual(original_track[1].age, 1, "Continuing fragment should have age 1")
        
        # Check that the other split fragment starts a new track
        # but inherits the age of the parent (age = 1)
        other_fragment_id = 2 if original_track[1].cloud_id == 3 else 3
        found_other_fragment = False
        
        for track_id, track in tracks.items():
            if track_id != original_track_id:
                for cloud in track:
                    if cloud.cloud_id == other_fragment_id:
                        found_other_fragment = True
                        self.assertEqual(len(track), 1, "New track for split should have 1 cloud")
                        self.assertEqual(cloud.age, 1, "Split fragment should inherit age 1")
        
        self.assertTrue(found_other_fragment, f"Could not find track containing cloud {other_fragment_id}")
        
        # Check that the new cloud starts a new track with age = 0
        found_new_cloud = False
        for track_id, track in tracks.items():
            for cloud in track:
                if cloud.cloud_id == 4:
                    found_new_cloud = True
                    self.assertEqual(cloud.age, 0, "New cloud should have age 0")
        
        self.assertTrue(found_new_cloud, "Could not find track containing the new cloud")

if __name__ == "__main__":
    unittest.main()