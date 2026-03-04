import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.cloudtracker import CloudTracker
from lib.cloud import Cloud


class MockCloudField:
    """Simple mock of CloudField for testing."""
    def __init__(self, clouds_dict, timestep=0):
        self.clouds = clouds_dict
        self.timestep = timestep


def _make_cloud(cloud_id, size, location, timestep, n_levels):
    """Helper: create a minimal Cloud for testing."""
    return Cloud(
        cloud_id=cloud_id,
        size=size,
        surface_area=size * 2,
        cloud_base_area=max(1, size // 2),
        cloud_base_height=location[2],
        location=location,
        points=[location],
        surface_points=np.array([location]),
        timestep=timestep,
        max_height=location[2],
        max_w=1.0,
        max_w_cloud_base=0.5,
        mean_u=np.zeros(n_levels),
        mean_v=np.zeros(n_levels),
        mean_w=np.zeros(n_levels),
        ql_flux=0.1,
        mass_flux=0.2,
        mass_flux_per_level=np.zeros(n_levels),
        temp_per_level=np.zeros(n_levels),
        theta_outside_per_level=np.zeros(n_levels),
        w_per_level=np.zeros(n_levels),
        circum_per_level=np.zeros(n_levels),
        eff_radius_per_level=np.zeros(n_levels),
    )


class TestCloudMergeAge(unittest.TestCase):
    """Test that merged clouds inherit the age of the oldest parent.

    Uses three timesteps so that clouds accumulate genuinely different ages
    before the merge, rather than relying on constructor-set ages that the
    tracker overwrites.
    """

    def setUp(self):
        self.config = {
            'min_size': 3,
            'timestep_duration': 60,
            'horizontal_resolution': 25.0,
            'switch_wind_drift': False,
            'switch_background_drift': False,
            'switch_vertical_drift': False,
            'max_expected_cloud_speed': 20.0,
            'bounding_box_safety_factor': 2.0,
            'use_pre_filtering': False,
        }
        self.zt = np.array([0, 100, 200, 300, 400, 500])
        self.xt = np.arange(0, 500, 25.0)
        self.yt = np.arange(0, 500, 25.0)
        self.n = len(self.zt)

    def test_merge_age_inheritance_default_criterion(self):
        """Oldest parent wins merge with default 'age' criterion.

        Timeline:
          T0: Cloud A (id=1, size=10)
          T1: A continues as A' (id=2), new Cloud B (id=3, size=8)
          T2: Both A' and B match merged Cloud C (id=4) -> merge.
              A' has age 1, B has age 0 -> A' wins.
              C inherits age 2 (=1+1).
        """
        tracker = CloudTracker(self.config)

        # Use a timestep-aware mock
        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                return last_cloud.cloud_id == 1 and cloud.cloud_id == 2
            if cf.timestep == 2:
                return cloud.cloud_id == 4 and last_cloud.cloud_id in [2, 3]
            return False

        tracker.is_match = mock_is_match

        # T0
        cloud_a = _make_cloud(1, 10, (100, 100, 200), 0, self.n)
        tracker.update_tracks(MockCloudField({1: cloud_a}, timestep=0),
                              self.zt, self.xt, self.yt)

        # T1
        cloud_a_prime = _make_cloud(2, 10, (102, 102, 210), 1, self.n)
        cloud_b = _make_cloud(3, 8, (300, 300, 250), 1, self.n)
        tracker.update_tracks(MockCloudField({2: cloud_a_prime, 3: cloud_b}, timestep=1),
                              self.zt, self.xt, self.yt)

        # Verify intermediate state
        self.assertEqual(tracker.cloud_tracks[1][-1].age, 1)
        self.assertEqual(tracker.cloud_tracks[3][-1].age, 0)

        # T2
        cloud_c = _make_cloud(4, 20, (120, 120, 220), 2, self.n)
        tracker.update_tracks(MockCloudField({4: cloud_c}, timestep=2),
                              self.zt, self.xt, self.yt)

        tracks = tracker.get_tracks()

        # Find the track that contains the merged cloud C
        merged_track_id = None
        for tid, track in tracks.items():
            if any(c.cloud_id == 4 for c in track):
                merged_track_id = tid
                break

        self.assertIsNotNone(merged_track_id)

        # The winning track should be track 1 (A -> A' -> C) because A' is older
        merged_track = tracks[merged_track_id]
        self.assertEqual(merged_track_id, 1)
        self.assertEqual(len(merged_track), 3)
        self.assertEqual(merged_track[-1].age, 2, "C should have age 2 (=1+1)")

        # Merged cloud records which tracks merged into it
        self.assertIn(3, merged_track[-1].merged_with)
        self.assertEqual(merged_track[-1].merges_count, 1)

        # Losing track B should be inactive with merged_into pointing to winner
        loser_track = tracks[3]
        self.assertFalse(loser_track[-1].is_active)
        self.assertEqual(loser_track[-1].merged_into, 1)


if __name__ == "__main__":
    unittest.main()
