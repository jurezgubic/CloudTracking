import unittest
import numpy as np
import sys
import os

# This test verifies:
# 1. Track Continuation: The LARGEST fragment continues the parent track (by cell count)
# 2. Split Inheritance: Other fragments start new tracks with age = parent + 1
# 3. Split Provenance: The continuation child does NOT carry split_from (it IS the parent)
# 4. New Cloud Handling: Genuinely new clouds start with age 0

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


class TestCloudSplitting(unittest.TestCase):
    """Test the cloud splitting behavior in CloudTracker."""

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

    def test_largest_child_continues_parent_track(self):
        """The largest child (by cell count) continues the parent track.

        T0: Cloud P (id=1, size=10)
        T1: Cloud big (id=2, size=6), Cloud small (id=3, size=4) both match P.
            Cloud new (id=4, size=8) matches nothing.

        Expected:
          - Track 1 = [P, big] (big is largest child, continues)
          - Track 3 = [small]  (new track, split_from=1)
          - Track 4 = [new]    (genuinely new, age=0)
        """
        tracker = CloudTracker(self.config)

        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                return last_cloud.cloud_id == 1 and cloud.cloud_id in [2, 3]
            return False

        tracker.is_match = mock_is_match

        # T0
        cloud_p = _make_cloud(1, 10, (100, 100, 200), 0, self.n)
        tracker.update_tracks(MockCloudField({1: cloud_p}, timestep=0),
                              self.zt, self.xt, self.yt)

        # T1
        cloud_big = _make_cloud(2, 6, (102, 102, 210), 1, self.n)
        cloud_small = _make_cloud(3, 4, (97, 99, 205), 1, self.n)
        cloud_new = _make_cloud(4, 8, (300, 300, 250), 1, self.n)
        tracker.update_tracks(
            MockCloudField({2: cloud_big, 3: cloud_small, 4: cloud_new}, timestep=1),
            self.zt, self.xt, self.yt,
        )

        tracks = tracker.get_tracks()

        # Parent track continues with the LARGER child (cloud_big, size=6)
        parent_track = tracks[1]
        self.assertEqual(len(parent_track), 2)
        self.assertEqual(parent_track[0].cloud_id, 1)
        self.assertEqual(parent_track[1].cloud_id, 2, "Largest child should continue")
        self.assertEqual(parent_track[1].age, 1)

        # Continuation child should NOT have split_from (it IS the parent track)
        self.assertIsNone(parent_track[1].split_from,
                          "Child continuing parent must not carry split_from")

        # Smaller child starts a new track with provenance
        split_track = tracks[3]
        self.assertEqual(len(split_track), 1)
        self.assertEqual(split_track[0].cloud_id, 3)
        self.assertEqual(split_track[0].age, 1)
        self.assertEqual(split_track[0].split_from, 1,
                         "Split child should reference parent track")

        # New cloud starts fresh
        new_track = tracks[4]
        self.assertEqual(len(new_track), 1)
        self.assertEqual(new_track[0].age, 0)
        self.assertIsNone(new_track[0].split_from)


if __name__ == "__main__":
    unittest.main()
