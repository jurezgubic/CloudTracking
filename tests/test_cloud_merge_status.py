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
    def __init__(self, clouds_dict, timestep=0):
        self.clouds = clouds_dict
        self.timestep = timestep

class TestCloudMergeStatus(unittest.TestCase):
    """Test that clouds are correctly marked as merged rather than died"""
    
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
        
        # Create mock height and horizontal levels
        self.zt = np.array([0, 100, 200, 300, 400, 500])
        self.xt = np.arange(0, 500, 25.0)  # x coordinates
        self.yt = np.arange(0, 500, 25.0)  # y coordinates
        
        # Create mean velocities for Cloud constructor
        self.mean_u = np.zeros_like(self.zt)
        self.mean_v = np.zeros_like(self.zt)
        self.mean_w = np.zeros_like(self.zt)
        
    def test_cloud_merge_status(self):
        """Test that clouds are correctly marked as merged rather than died"""
        # Initialize tracker
        tracker = CloudTracker(self.config)
        
        # TIMESTEP 1: Create two clouds with different ages
        cloud1 = Cloud(
            cloud_id=1,
            size=10,
            surface_area=20,
            cloud_base_area=5,
            cloud_base_height=200,
            location=(100, 100, 200),
            points=[(100, 100, 200), (101, 100, 200)],
            surface_points=np.array([(100, 100, 200)]),
            timestep=0,
            max_height=200,
            max_w=1.0,
            max_w_cloud_base=0.5,
            mean_u=self.mean_u,
            mean_v=self.mean_v,
            mean_w=self.mean_w,
            ql_flux=0.1,
            mass_flux=0.2,
            mass_flux_per_level=np.zeros_like(self.zt),
            temp_per_level=np.zeros_like(self.zt),
            theta_outside_per_level=np.zeros_like(self.zt),
            w_per_level=np.zeros_like(self.zt),
            circum_per_level=np.zeros_like(self.zt),
            eff_radius_per_level=np.zeros_like(self.zt),
            age=2  # This cloud is older
        )
        
        cloud2 = Cloud(
            cloud_id=2,
            size=8,
            surface_area=16,
            cloud_base_area=4,
            cloud_base_height=220,
            location=(150, 150, 220),
            points=[(150, 150, 220), (151, 150, 220)],
            surface_points=np.array([(150, 150, 220)]),
            timestep=0,
            max_height=220,
            max_w=1.2,
            max_w_cloud_base=0.6,
            mean_u=self.mean_u,
            mean_v=self.mean_v,
            mean_w=self.mean_w,
            ql_flux=0.12,
            mass_flux=0.22,
            mass_flux_per_level=np.zeros_like(self.zt),
            temp_per_level=np.zeros_like(self.zt),
            theta_outside_per_level=np.zeros_like(self.zt),
            w_per_level=np.zeros_like(self.zt),
            circum_per_level=np.zeros_like(self.zt),
            eff_radius_per_level=np.zeros_like(self.zt),
            age=1  # This cloud is younger
        )
        
        # Add the clouds to a mock CloudField
        cloud_field1 = MockCloudField({1: cloud1, 2: cloud2})
        
        # Update tracker with the first cloud field
        tracker.update_tracks(cloud_field1, self.zt, self.xt, self.yt)
        
        # TIMESTEP 2: Create a merged cloud
        merged_cloud = Cloud(
            cloud_id=3,
            size=18,  # Sum of the two original clouds
            surface_area=36,
            cloud_base_area=9,
            cloud_base_height=230,
            location=(125, 125, 230),  # Between the two original clouds
            points=[(125, 125, 230), (126, 125, 230)],
            surface_points=np.array([(125, 125, 230)]),
            timestep=1,
            max_height=230,
            max_w=1.3,
            max_w_cloud_base=0.7,
            mean_u=self.mean_u,
            mean_v=self.mean_v,
            mean_w=self.mean_w,
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
        cloud_field2 = MockCloudField({3: merged_cloud})
        
        # Override is_match method to force the merge
        original_is_match = tracker.is_match
        
        def mock_is_match(cloud, last_cloud_in_track, current_cloud_field):
            # Only return True for specific matches we want to test
            # cloud is the current cloud (merged_cloud in this case)
            # last_cloud_in_track is one of the original clouds (cloud1 or cloud2)
            
            # Only match merged_cloud (id=3) with original clouds (id=1 or id=2)
            if cloud.cloud_id == 3 and last_cloud_in_track.cloud_id in [1, 2]:
                return True
            return False
        
        # Apply mock method
        tracker.is_match = mock_is_match
        
        # Update tracker with the second cloud field
        tracker.update_tracks(cloud_field2, self.zt, self.xt, self.yt)
        
        # Restore original is_match method
        tracker.is_match = original_is_match
        
        # Get all tracks
        tracks = tracker.get_tracks()
        
        # Find the tracks containing our clouds
        track1 = None
        track2 = None
        merged_track_id = None
        
        for track_id, track in tracks.items():
            if any(c.cloud_id == 1 for c in track):
                track1 = track
                if len(track) > 1:  # If this track continued
                    merged_track_id = track_id
            elif any(c.cloud_id == 2 for c in track):
                track2 = track
        
        # One track should continue (the one with the older cloud)
        self.assertIsNotNone(track1, "Could not find track containing cloud1")
        self.assertIsNotNone(track2, "Could not find track containing cloud2")
        
        # Get the last cloud in each track
        last_cloud1 = track1[-1]
        last_cloud2 = track2[-1]
        
        # Instead of hard-coding expectations based on cloud_id values
        # We could check that the track with the older cloud continues:

        # Find which track continued (has the merged cloud)
        continued_track = None
        continued_track_id = None
        for track_id, track in tracks.items():
            if any(c.cloud_id == 3 for c in track):
                continued_track = track
                continued_track_id = track_id
                break

        self.assertIsNotNone(continued_track, "No track contains the merged cloud")

        # Find the original cloud in the continued track
        original_cloud_id = continued_track[0].cloud_id

        # Check that the continued track has the older cloud
        self.assertEqual(
            original_cloud_id, 
            1, 
            "The track of the older cloud should continue with the merged cloud"
        )

        # Check the other track ended properly
        other_track_id = 2 if original_cloud_id == 1 else 1
        other_track = None
        for track_id, track in tracks.items():
            if any(c.cloud_id == other_track_id for c in track):
                other_track = track
                break

        self.assertIsNotNone(other_track, f"Could not find track with cloud {other_track_id}")
        self.assertEqual(len(other_track), 1, "The younger cloud's track should end")
        self.assertFalse(other_track[0].is_active, "The younger cloud should be marked inactive")
        self.assertEqual(other_track[0].merged_into, continued_track_id, 
                       "merged_into should point to the track the cloud merged into")

if __name__ == "__main__":
    unittest.main()