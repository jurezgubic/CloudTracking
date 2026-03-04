"""
Tests for cloud merge age inheritance.

Verifies that merged clouds inherit the age of the oldest parent (default
criterion). Uses three timesteps so that clouds accumulate genuinely different
ages before the merge, rather than relying on constructor-set ages that the
tracker overwrites.
"""

import numpy as np
from lib.cloudtracker import CloudTracker
from tests.conftest import make_cloud, SimpleMockCloudField


class TestCloudMergeAge:
    """Merged clouds inherit the age of the oldest parent."""

    def test_merge_age_inheritance_default_criterion(
            self, tracker_config, domain_grids):
        """Oldest parent wins merge with default age criterion.

        Timeline:
          T0: Cloud A (id=1, size=10)
          T1: A continues as A' (id=2), new Cloud B (id=3, size=8)
          T2: Both A' and B match merged Cloud C (id=4) -> merge.
              A' has age 1, B has age 0 -> A' wins.
              C inherits age 2 (=1+1).
        """
        zt, xt, yt = domain_grids
        n = len(zt)
        tracker = CloudTracker(tracker_config)

        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                return last_cloud.cloud_id == 1 and cloud.cloud_id == 2
            if cf.timestep == 2:
                return cloud.cloud_id == 4 and last_cloud.cloud_id in [2, 3]
            return False

        tracker.is_match = mock_is_match

        # T0
        cloud_a = make_cloud(1, 10, (100, 100, 200), 0, n)
        tracker.update_tracks(
            SimpleMockCloudField({1: cloud_a}, timestep=0),
            zt, xt, yt)

        # T1
        cloud_a_prime = make_cloud(2, 10, (102, 102, 210), 1, n)
        cloud_b = make_cloud(3, 8, (300, 300, 250), 1, n)
        tracker.update_tracks(
            SimpleMockCloudField({2: cloud_a_prime, 3: cloud_b}, timestep=1),
            zt, xt, yt)

        # Verify intermediate state
        assert tracker.cloud_tracks[1][-1].age == 1
        assert tracker.cloud_tracks[3][-1].age == 0

        # T2
        cloud_c = make_cloud(4, 20, (120, 120, 220), 2, n)
        tracker.update_tracks(
            SimpleMockCloudField({4: cloud_c}, timestep=2),
            zt, xt, yt)

        tracks = tracker.get_tracks()

        # Find the track that contains the merged cloud C
        merged_track_id = None
        for tid, track in tracks.items():
            if any(c.cloud_id == 4 for c in track):
                merged_track_id = tid
                break

        assert merged_track_id is not None

        # The winning track should be track 1 (A -> A' -> C) because A' is older
        merged_track = tracks[merged_track_id]
        assert merged_track_id == 1
        assert len(merged_track) == 3
        assert merged_track[-1].age == 2, "C should have age 2 (=1+1)"

        # Merged cloud records which tracks merged into it
        assert 3 in merged_track[-1].merged_with
        assert merged_track[-1].merges_count == 1

        # Losing track B should be inactive with merged_into pointing to winner
        loser_track = tracks[3]
        assert not loser_track[-1].is_active
        assert loser_track[-1].merged_into == 1
