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


class TestCloudMergeStatus(unittest.TestCase):
    """Test that merge losers are correctly marked (inactive + merged_into)."""

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

    def test_merge_loser_marked_correctly(self):
        """Merge loser should be inactive with merged_into pointing to winner.

        T0: Cloud A (id=1, size=10), Cloud B (id=2, size=8)
        T1: Cloud M (id=3) matched by both -> merge.
            A wins (same age=0, larger size tiebreaker). B loses.
        """
        tracker = CloudTracker(self.config)

        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                return cloud.cloud_id == 3 and last_cloud.cloud_id in [1, 2]
            return False

        tracker.is_match = mock_is_match

        # T0
        cloud_a = _make_cloud(1, 10, (100, 100, 200), 0, self.n)
        cloud_b = _make_cloud(2, 8, (150, 150, 220), 0, self.n)
        tracker.update_tracks(MockCloudField({1: cloud_a, 2: cloud_b}, timestep=0),
                              self.zt, self.xt, self.yt)

        # T1
        merged = _make_cloud(3, 18, (125, 125, 230), 1, self.n)
        tracker.update_tracks(MockCloudField({3: merged}, timestep=1),
                              self.zt, self.xt, self.yt)

        tracks = tracker.get_tracks()

        # Find which track continued (contains merged cloud)
        continued_track_id = None
        for tid, track in tracks.items():
            if any(c.cloud_id == 3 for c in track):
                continued_track_id = tid
                break

        self.assertIsNotNone(continued_track_id)
        # A should win (same age, bigger size)
        self.assertEqual(continued_track_id, 1)

        continued_track = tracks[continued_track_id]
        self.assertEqual(len(continued_track), 2)
        self.assertEqual(continued_track[-1].cloud_id, 3)
        self.assertEqual(continued_track[-1].merges_count, 1)
        self.assertIn(2, continued_track[-1].merged_with)

        # B's track should end: inactive + merged_into
        loser_track = tracks[2]
        self.assertEqual(len(loser_track), 1)
        self.assertFalse(loser_track[-1].is_active)
        self.assertEqual(loser_track[-1].merged_into, continued_track_id)


if __name__ == "__main__":
    unittest.main()
