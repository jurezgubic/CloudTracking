import unittest
import numpy as np
import sys
import os

# Add parent directory to path to import from lib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.cloudtracker import CloudTracker
from lib.cloud import Cloud

class MockCloudField:
    """Simple mock of CloudField for testing"""
    def __init__(self, clouds_dict):
        self.clouds = clouds_dict

class TestCloudMergeAge(unittest.TestCase):
    """Test that merged clouds inherit the age of the oldest parent"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = {
            'min_size': 3,
            'timestep_duration': 60,
            'horizontal_resolution': 25.0,
            'switch_wind_drift': False,
            'switch_background_drift': False,
            'switch_vertical_drift': False
        }
        
        # Create mock height levels
        self.zt = np.array([0, 100, 200, 300, 400, 500])
        
        # Create mean velocities (all zeros for simplicity)
        self.mean_u = np.zeros_like(self.zt)
        self.mean_v = np.zeros_like(self.zt)
        self.mean_w = np.zeros_like(self.zt)
        
    def test_merge_age_inheritance(self):
        """Test that merged clouds inherit the age of the oldest parent"""
        # Initialize tracker
        tracker = CloudTracker(self.config)
        
        # TIMESTEP 1: Create three clouds with different ages
        cloud1 = Cloud(
            cloud_id=1,
            size=10,
            surface_area=20,
            cloud_base_area=5,
            location=(100, 100, 200),
            points=[(100, 100, 200), (101, 100, 200)],
            timestep=0,
            max_height=200,
            max_w=1.0,
            max_w_cloud_base=0.5,
            ql_flux=0.1,
            mass_flux=0.2,
            mass_flux_per_level=np.zeros_like(self.zt),
            temp_per_level=np.zeros_like(self.zt),
            theta_outside_per_level=np.zeros_like(self.zt),
            w_per_level=np.zeros_like(self.zt),
            circum_per_level=np.zeros_like(self.zt),
            eff_radius_per_level=np.zeros_like(self.zt),
            age=5  # This cloud is oldest
        )
        
        cloud2 = Cloud(
            cloud_id=2,
            size=8,
            surface_area=16,
            cloud_base_area=4,
            location=(150, 150, 220),
            points=[(150, 150, 220), (151, 150, 220)],
            timestep=0,
            max_height=220,
            max_w=1.2,
            max_w_cloud_base=0.6,
            ql_flux=0.12,
            mass_flux=0.22,
            mass_flux_per_level=np.zeros_like(self.zt),
            temp_per_level=np.zeros_like(self.zt),
            theta_outside_per_level=np.zeros_like(self.zt),
            w_per_level=np.zeros_like(self.zt),
            circum_per_level=np.zeros_like(self.zt),
            eff_radius_per_level=np.zeros_like(self.zt),
            age=3  # This cloud is middle-aged
        )
        
        cloud3 = Cloud(
            cloud_id=3,
            size=6,
            surface_area=12,
            cloud_base_area=3,
            location=(175, 175, 210),
            points=[(175, 175, 210), (176, 175, 210)],
            timestep=0,
            max_height=210, 
            max_w=0.9,
            max_w_cloud_base=0.4,
            ql_flux=0.08,
            mass_flux=0.18,
            mass_flux_per_level=np.zeros_like(self.zt),
            temp_per_level=np.zeros_like(self.zt),
            theta_outside_per_level=np.zeros_like(self.zt),
            w_per_level=np.zeros_like(self.zt),
            circum_per_level=np.zeros_like(self.zt),
            eff_radius_per_level=np.zeros_like(self.zt),
            age=1  # This cloud is youngest
        )
        
        # Add the clouds to a mock CloudField
        cloud_field1 = MockCloudField({1: cloud1, 2: cloud2, 3: cloud3})
        
        # Update tracker with the first cloud field
        tracker.update_tracks(cloud_field1, self.mean_u, self.mean_v, self.mean_w, self.zt)
        
        # TIMESTEP 2: Create a merged cloud
        merged_cloud = Cloud(
            cloud_id=4,
            size=24,  # Sum of the three original clouds
            surface_area=48,
            cloud_base_area=12,
            location=(140, 140, 230),  # Somewhere in the middle
            points=[(140, 140, 230), (141, 140, 230)],
            timestep=1,
            max_height=230,
            max_w=1.3,
            max_w_cloud_base=0.7,
            ql_flux=0.13,
            mass_flux=0.23,
            mass_flux_per_level=np.zeros_like(self.zt),
            temp_per_level=np.zeros_like(self.zt),
            theta_outside_per_level=np.zeros_like(self.zt),
            w_per_level=np.zeros_like(self.zt),
            circum_per_level=np.zeros_like(self.zt),
            eff_radius_per_level=np.zeros_like(self.zt)
        )
        
        # Add the merged cloud to a mock CloudField
        cloud_field2 = MockCloudField({4: merged_cloud})
        
        # Override is_match method to force the merge
        original_is_match = tracker.is_match
        
        def mock_is_match(cloud, last_cloud_in_track):
            # All three clouds should match with merged_cloud
            return True
        
        # Apply mock method
        tracker.is_match = mock_is_match
        
        # Update tracker with the second cloud field
        tracker.update_tracks(cloud_field2, self.mean_u, self.mean_v, self.mean_w, self.zt)
        
        # Restore original is_match method
        tracker.is_match = original_is_match
        
        # Get all tracks
        tracks = tracker.get_tracks()
        
        # Find which track contains the merged cloud
        merged_track = None
        merged_track_id = None
        for track_id, track in tracks.items():
            if any(c.cloud_id == 4 for c in track):
                merged_track = track
                merged_track_id = track_id
                break
        
        self.assertIsNotNone(merged_track, "Could not find track containing the merged cloud")
        
        # Find the parent cloud in the merged track
        parent_in_merged_track = merged_track[0]
        
        # The parent in the merged track should be the oldest cloud (cloud1)
        self.assertEqual(parent_in_merged_track.cloud_id, 1, 
                       "The oldest cloud should be the parent in the merged track")
        
        # The merged cloud should have age = oldest parent's age + 1
        self.assertEqual(merged_track[-1].age, cloud1.age + 1, 
                       "Merged cloud should inherit age from oldest parent + 1")
        
        # The other clouds should have merged_into pointing to the merged track
        for track_id, track in tracks.items():
            if track_id != merged_track_id:
                last_cloud = track[-1]
                if last_cloud.cloud_id in [2, 3]:  # These were the other clouds
                    self.assertFalse(last_cloud.is_active, f"Cloud {last_cloud.cloud_id} should be inactive")
                    self.assertEqual(last_cloud.merged_into, merged_track_id, 
                                  f"Cloud {last_cloud.cloud_id} should have merged_into pointing to track {merged_track_id}")

if __name__ == "__main__":
    unittest.main()