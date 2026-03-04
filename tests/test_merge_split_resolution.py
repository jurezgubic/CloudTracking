"""Tests for merge/split resolution edge cases (Issues 1, 3, 4, 6)."""

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


class TestStaleMergedIntoClearedOnSplitContinuation(unittest.TestCase):
    """Issue 1: When a merge loser has another child (simultaneous split+merge),
    the loser's track continues.  The stale merged_into must be cleared.

    T0: Cloud A (id=1, size=10), Cloud B (id=2, size=8)
    T1: Cloud M (id=3) matched by both A and B -> merge (A wins).
        Cloud S (id=4) matched by B only -> B's split child.

    After resolution:
      - Track 1 = [A, M]  (merge winner)
      - Track 2 = [B, S]  (B lost the merge but survived via S)
      - B.merged_into should be None (cleared), because B's track did not end.
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

    def test_merged_into_cleared_when_track_survives(self):
        tracker = CloudTracker(self.config)

        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                # M (id=3) matched by both A (id=1) and B (id=2)
                if cloud.cloud_id == 3 and last_cloud.cloud_id in [1, 2]:
                    return True
                # S (id=4) matched by B (id=2) only
                if cloud.cloud_id == 4 and last_cloud.cloud_id == 2:
                    return True
            return False

        tracker.is_match = mock_is_match

        # T0
        cloud_a = _make_cloud(1, 10, (100, 100, 200), 0, self.n)
        cloud_b = _make_cloud(2, 8, (200, 200, 200), 0, self.n)
        tracker.update_tracks(MockCloudField({1: cloud_a, 2: cloud_b}, timestep=0),
                              self.zt, self.xt, self.yt)

        # T1
        cloud_m = _make_cloud(3, 18, (110, 110, 210), 1, self.n)
        cloud_s = _make_cloud(4, 6, (210, 210, 210), 1, self.n)
        tracker.update_tracks(
            MockCloudField({3: cloud_m, 4: cloud_s}, timestep=1),
            self.zt, self.xt, self.yt,
        )

        tracks = tracker.get_tracks()

        # Track 1 won the merge: [A, M]
        self.assertEqual(len(tracks[1]), 2)
        self.assertEqual(tracks[1][-1].cloud_id, 3)

        # Track 2 survived via S: [B, S]
        self.assertEqual(len(tracks[2]), 2)
        self.assertEqual(tracks[2][-1].cloud_id, 4)
        self.assertTrue(tracks[2][-1].is_active)

        # B's merged_into was set in Pass 2 but must be cleared in Pass 3
        cloud_b_in_track = tracks[2][0]
        self.assertIsNone(cloud_b_in_track.merged_into,
                          "merged_into must be cleared when track survives via split child")

        # M should still record that track 2 merged into it
        self.assertIn(2, tracks[1][-1].merged_with)


class TestMergeWinnerSizeCriterion(unittest.TestCase):
    """Issue 3: With merge_winner_criterion='size', the largest parent wins
    even if a smaller parent is older.

    T0: Cloud A (id=1, size=5)
    T1: A continues as A' (id=2, size=5), new Cloud B (id=3, size=20)
    T2: Cloud C (id=4) matched by both A' and B.
        With 'age': A' wins (age=1 > 0).
        With 'size': B wins (size=20 > 5).
    """

    def setUp(self):
        self.zt = np.array([0, 100, 200, 300, 400, 500])
        self.xt = np.arange(0, 500, 25.0)
        self.yt = np.arange(0, 500, 25.0)
        self.n = len(self.zt)
        self.base_config = {
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

    def _run_merge(self, criterion):
        config = {**self.base_config, 'merge_winner_criterion': criterion}
        tracker = CloudTracker(config)

        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                return last_cloud.cloud_id == 1 and cloud.cloud_id == 2
            if cf.timestep == 2:
                return cloud.cloud_id == 4 and last_cloud.cloud_id in [2, 3]
            return False

        tracker.is_match = mock_is_match

        # T0: small cloud A
        tracker.update_tracks(
            MockCloudField({1: _make_cloud(1, 5, (100, 100, 200), 0, self.n)}, timestep=0),
            self.zt, self.xt, self.yt,
        )
        # T1: A continues, big cloud B appears
        tracker.update_tracks(
            MockCloudField({
                2: _make_cloud(2, 5, (102, 102, 210), 1, self.n),
                3: _make_cloud(3, 20, (300, 300, 250), 1, self.n),
            }, timestep=1),
            self.zt, self.xt, self.yt,
        )
        # T2: merge
        tracker.update_tracks(
            MockCloudField({4: _make_cloud(4, 25, (120, 120, 220), 2, self.n)}, timestep=2),
            self.zt, self.xt, self.yt,
        )
        return tracker.get_tracks()

    def test_age_criterion_oldest_wins(self):
        tracks = self._run_merge('age')
        # Track 1 (A -> A' -> C) should win: A' has age 1 > B's age 0
        self.assertIn(4, [c.cloud_id for c in tracks[1]])
        self.assertEqual(tracks[1][-1].age, 2)

    def test_size_criterion_largest_wins(self):
        tracks = self._run_merge('size')
        # Track 3 (B -> C) should win: B has size 20 > A' size 5
        self.assertIn(4, [c.cloud_id for c in tracks[3]])
        self.assertEqual(tracks[3][-1].age, 1)  # B age was 0, C = 0+1

        # Track 1 (A -> A') should be the loser
        self.assertFalse(tracks[1][-1].is_active)
        self.assertEqual(tracks[1][-1].merged_into, 3)


class TestDowngradedMergeOrphanProvenance(unittest.TestCase):
    """Issue 4: When all parents of a merge candidate are already committed,
    the orphaned cloud should still receive split_from provenance.

    To trigger the all-parents-committed path, M1 and M2 must be processed
    BEFORE Orphan.  sorted_merge_ids orders by parent count (descending),
    so giving M1 and M2 three parents each (vs two for Orphan) guarantees
    they are resolved first and commit tracks A and B.

    T0: A (id=1, size=30), B (id=2, size=25), X (id=3, size=15), Y (id=4, size=10)
    T1: M1 (id=5) matched by A, X, Y  -> 3 parents -> A wins, committed.
        M2 (id=6) matched by B, X, Y  -> 3 parents -> B wins, committed.
        Orphan (id=7) matched by A, B -> 2 parents -> both committed -> downgraded.

    Expected: Orphan starts a new track with split_from set.
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

    def test_orphan_gets_split_from(self):
        tracker = CloudTracker(self.config)

        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                # M1 (id=5) matched by A (id=1), X (id=3), Y (id=4)
                if cloud.cloud_id == 5 and last_cloud.cloud_id in [1, 3, 4]:
                    return True
                # M2 (id=6) matched by B (id=2), X (id=3), Y (id=4)
                if cloud.cloud_id == 6 and last_cloud.cloud_id in [2, 3, 4]:
                    return True
                # Orphan (id=7) matched by A (id=1) and B (id=2)
                if cloud.cloud_id == 7 and last_cloud.cloud_id in [1, 2]:
                    return True
            return False

        tracker.is_match = mock_is_match

        # T0
        tracker.update_tracks(MockCloudField({
            1: _make_cloud(1, 30, (100, 100, 200), 0, self.n),
            2: _make_cloud(2, 25, (200, 200, 200), 0, self.n),
            3: _make_cloud(3, 15, (300, 300, 200), 0, self.n),
            4: _make_cloud(4, 10, (400, 400, 200), 0, self.n),
        }, timestep=0), self.zt, self.xt, self.yt)

        # T1
        tracker.update_tracks(MockCloudField({
            5: _make_cloud(5, 40, (110, 110, 210), 1, self.n),
            6: _make_cloud(6, 35, (210, 210, 210), 1, self.n),
            7: _make_cloud(7, 8, (150, 150, 210), 1, self.n),
        }, timestep=1), self.zt, self.xt, self.yt)

        tracks = tracker.get_tracks()

        # M1 (id=5) should be in track 1 (A won, biggest)
        self.assertIn(5, [c.cloud_id for c in tracks[1]])
        # M2 (id=6) should be in track 2 (B won, biggest)
        self.assertIn(6, [c.cloud_id for c in tracks[2]])

        # Orphan (id=7) starts a new track with provenance
        orphan_track = tracks[7]
        self.assertEqual(len(orphan_track), 1)
        self.assertEqual(orphan_track[0].cloud_id, 7)
        self.assertIsNotNone(orphan_track[0].split_from,
                             "Orphaned merge child must have split_from provenance")
        self.assertIn(orphan_track[0].split_from, [1, 2],
                      "split_from should point to one of the committed parents")


if __name__ == "__main__":
    unittest.main()
