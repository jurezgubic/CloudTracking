"""
Tests for cloud track creation.

These tests verify that CloudTracker correctly creates new tracks
and handles the update_tracks() workflow.
"""
import unittest
import numpy as np
from lib.cloudtracker import CloudTracker
from lib.cloud import Cloud
from tests.test_utils import MockCloudField


class TestCloudTracker(unittest.TestCase):
    """Test the CloudTracker class track creation."""
    
    def setUp(self):
        """Set up the test case."""
        # Basic configuration for testing
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
        self.cloud_tracker = CloudTracker(self.config)
        
        # Set up domain info
        self.zt = np.linspace(0, 1000, 10)
        self.xt = np.linspace(0, 1000, 41)  # 25m resolution
        self.yt = np.linspace(0, 1000, 41)
        
        # Initialize tracker domain info
        self.cloud_tracker.zt = self.zt
        self.cloud_tracker.xt = self.xt
        self.cloud_tracker.yt = self.yt
        self.cloud_tracker.domain_size_x = 1000.0
        self.cloud_tracker.domain_size_y = 1000.0
        
    def create_mock_cloud(self, cloud_id, x, y, z, is_active=True, age=0):
        """Create a mock cloud for testing."""
        # Create surface points around the center
        surface_points = np.array([
            [x, y, z],
            [x+10, y, z],
            [x, y+10, z],
            [x-10, y, z],
            [x, y-10, z],
        ], dtype=np.float32)
        
        return Cloud(
            cloud_id=cloud_id,
            size=10,
            surface_area=5,
            cloud_base_area=5,
            cloud_base_height=z,
            location=(x, y, z),
            points=[(x, y, z)],
            surface_points=surface_points,
            timestep=0,
            max_height=z,
            max_w=1.0,
            max_w_cloud_base=0.5,
            mean_u=0.0,  # Scalar values for is_match
            mean_v=0.0,
            mean_w=0.0,
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
        
    def test_inactive_cloud_doesnt_match_new_cloud(self):
        """Test that new clouds don't get matched with inactive clouds."""
        # Timestep 1: Cloud A appears
        cloud_a_t1 = self.create_mock_cloud(1, 100, 100, 500)
        cloud_field_t1 = MockCloudField({cloud_a_t1.cloud_id: cloud_a_t1})
        
        # Process first timestep - should create a new track
        self.cloud_tracker.update_tracks(cloud_field_t1, self.zt, self.xt, self.yt)
        
        # Timestep 2: Cloud A at same location (should match)
        cloud_a_t2 = self.create_mock_cloud(2, 100, 100, 500, age=1)
        cloud_field_t2 = MockCloudField({cloud_a_t2.cloud_id: cloud_a_t2})
        
        # Process second timestep - should match and add to track
        self.cloud_tracker.update_tracks(cloud_field_t2, self.zt, self.xt, self.yt)
        
        # Verify Cloud A has a track with 2 timesteps
        track_ids = list(self.cloud_tracker.cloud_tracks.keys())
        self.assertEqual(len(track_ids), 1, "Should have exactly one track")
        track_a = self.cloud_tracker.cloud_tracks[track_ids[0]]
        self.assertEqual(len(track_a), 2, "Track A should have 2 clouds")
        
        # Mark Cloud A as inactive
        track_a[-1].is_active = False
        
        # Timestep 3: Cloud A is gone, Cloud B appears far away
        cloud_b_t3 = self.create_mock_cloud(3, 500, 500, 400)  # Different location
        cloud_field_t3 = MockCloudField({cloud_b_t3.cloud_id: cloud_b_t3})
        
        # Process third timestep
        self.cloud_tracker.update_tracks(cloud_field_t3, self.zt, self.xt, self.yt)
        
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

    def test_first_cloud_creates_new_track(self):
        """Test that the first cloud creates a new track."""
        cloud = self.create_mock_cloud(1, 100, 100, 500)
        cloud_field = MockCloudField({cloud.cloud_id: cloud})
        
        self.cloud_tracker.update_tracks(cloud_field, self.zt, self.xt, self.yt)
        
        self.assertEqual(len(self.cloud_tracker.cloud_tracks), 1,
                         "Should have exactly one track after first cloud")
        
    def test_matched_cloud_extends_track(self):
        """Test that a matched cloud extends an existing track."""
        # First cloud
        cloud_t1 = self.create_mock_cloud(1, 100, 100, 500)
        cloud_field_t1 = MockCloudField({cloud_t1.cloud_id: cloud_t1})
        self.cloud_tracker.update_tracks(cloud_field_t1, self.zt, self.xt, self.yt)
        
        # Second cloud at same location (overlapping surface points)
        cloud_t2 = self.create_mock_cloud(2, 100, 100, 500)
        cloud_field_t2 = MockCloudField({cloud_t2.cloud_id: cloud_t2})
        self.cloud_tracker.update_tracks(cloud_field_t2, self.zt, self.xt, self.yt)
        
        # Should still have 1 track with 2 clouds
        self.assertEqual(len(self.cloud_tracker.cloud_tracks), 1,
                         "Should still have exactly one track")
        track = list(self.cloud_tracker.cloud_tracks.values())[0]
        self.assertEqual(len(track), 2, "Track should have 2 clouds")


if __name__ == '__main__':
    unittest.main()