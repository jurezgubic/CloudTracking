import unittest
import numpy as np
from lib.cloudtracker import CloudTracker
from lib.cloud import Cloud

class TestCloudTracker(unittest.TestCase):
    """Test the CloudTracker class."""
    
    def setUp(self):
        """Set up the test case."""
        # Basic configuration for testing
        self.config = {
            'horizontal_resolution': 25.0,
            'timestep_duration': 60,
            'switch_background_drift': False,
            'switch_wind_drift': False,
            'distance_threshold': 3,
        }
        self.cloud_tracker = CloudTracker(self.config)
        
        # Mock mean_u, mean_v, and zt for the test
        self.mean_u = np.zeros(10)
        self.mean_v = np.zeros(10)
        self.zt = np.linspace(0, 1000, 10)
        
    def create_mock_cloud(self, cloud_id, x, y, z, is_active=True, age=0):
        """Create a mock cloud for testing."""
        points = [(x, y, z)]
        return Cloud(
            cloud_id=cloud_id,
            size=10,
            surface_area=5,
            cloud_base_area=5,
            location=(x, y, z),
            points=points,
            timestep=0,
            max_height=z,
            max_w=1.0,
            max_w_cloud_base=0.5,
            ql_flux=0.1,
            mass_flux=0.2,
            mass_flux_per_level=np.zeros(10),
            temp_per_level=np.zeros(10),
            theta_outside_per_level=np.zeros(10),
            w_per_level=np.zeros(10),
            circum_per_level=np.zeros(10),
            eff_radius_per_level=np.zeros(10),
            is_active=is_active,
            age=age
        )
        
    def create_mock_cloud_field(self, clouds):
        """Create a mock cloud field for testing."""
        class MockCloudField:
            def __init__(self, clouds_dict):
                self.clouds = clouds_dict
        
        return MockCloudField({cloud.cloud_id: cloud for cloud in clouds})
        
    def test_inactive_cloud_doesnt_match_new_cloud(self):
        """Test that new clouds don't get matched with inactive clouds."""
        # Timestep 1: Cloud A appears
        cloud_a_t1 = self.create_mock_cloud(1, 100, 100, 500)
        cloud_field_t1 = self.create_mock_cloud_field([cloud_a_t1])
        
        # Process first timestep - should create a new track
        self.cloud_tracker.update_tracks(cloud_field_t1, self.mean_u, self.mean_v, self.zt)
        
        # Timestep 2: Cloud A moves slightly
        cloud_a_t2 = self.create_mock_cloud(2, 110, 110, 550, age=1)
        cloud_field_t2 = self.create_mock_cloud_field([cloud_a_t2])
        
        # Process second timestep - should match and add to track
        self.cloud_tracker.update_tracks(cloud_field_t2, self.mean_u, self.mean_v, self.zt)
        
        # Verify Cloud A has a track with 2 timesteps
        track_ids = list(self.cloud_tracker.cloud_tracks.keys())
        self.assertEqual(len(track_ids), 1, "Should have exactly one track")
        track_a = self.cloud_tracker.cloud_tracks[track_ids[0]]
        self.assertEqual(len(track_a), 2, "Track A should have 2 clouds")
        
        # Mark Cloud A as inactive
        track_a[-1].is_active = False
        
        # Timestep 3: Cloud A is gone, Cloud B appears far away
        cloud_b_t3 = self.create_mock_cloud(3, 500, 500, 400)  # Different location
        cloud_field_t3 = self.create_mock_cloud_field([cloud_b_t3])
        
        # Process third timestep
        self.cloud_tracker.update_tracks(cloud_field_t3, self.mean_u, self.mean_v, self.zt)
        
        # Verify results - should have 2 tracks now
        self.assertEqual(len(self.cloud_tracker.cloud_tracks), 2, 
                         "Should have two tracks: one for Cloud A and one for Cloud B")
                         
        # Check that Cloud B is in its own track and not added to Cloud A's track
        track_a = self.cloud_tracker.cloud_tracks[track_ids[0]]
        self.assertEqual(len(track_a), 2, 
                         "Track A should still have only 2 clouds (Cloud B wasn't added)")
                         
        # Find Cloud B's track
        track_b_id = None
        for track_id, track in self.cloud_tracker.cloud_tracks.items():
            if track_id != track_ids[0]:  # Not Track A
                track_b_id = track_id
                break
        
        self.assertIsNotNone(track_b_id, "Should have a track ID for Cloud B")
        track_b = self.cloud_tracker.cloud_tracks[track_b_id]
        self.assertEqual(len(track_b), 1, "Track B should have 1 cloud")
        self.assertEqual(track_b[0].cloud_id, 3, "Track B should contain Cloud B")
        self.assertEqual(track_b[0].age, 0, "Cloud B should have age 0")

if __name__ == '__main__':
    unittest.main()