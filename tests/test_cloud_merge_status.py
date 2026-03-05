"""
Tests for cloud merge loser status.

Verifies that when two same-age clouds merge, the larger one wins (size
tiebreaker). The loser is correctly marked inactive with merged_into
pointing to the winner track.
"""

import numpy as np
from lib.cloudtracker import CloudTracker
from tests.conftest import make_cloud, SimpleMockCloudField


class TestCloudMergeStatus:
    """Merge loser is correctly marked (inactive + merged_into)."""

    def test_merge_loser_marked_correctly(
            self, tracker_config, domain_grids):
        """T0: Cloud A (id=1, size=10), Cloud B (id=2, size=8)
        T1: Cloud M (id=3) matched by both -> merge.
            A wins (same age=0, larger size). B loses.
        """
        zt, xt, yt = domain_grids
        n = len(zt)
        tracker = CloudTracker(tracker_config)

        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                return cloud.cloud_id == 3 and last_cloud.cloud_id in [1, 2]
            return False

        tracker.is_match = mock_is_match

        # T0
        cloud_a = make_cloud(1, 10, (100, 100, 200), 0, n)
        cloud_b = make_cloud(2, 8, (150, 150, 220), 0, n)
        tracker.update_tracks(
            SimpleMockCloudField({1: cloud_a, 2: cloud_b}, timestep=0),
            zt, xt, yt)

        # T1
        merged = make_cloud(3, 18, (125, 125, 230), 1, n)
        tracker.update_tracks(
            SimpleMockCloudField({3: merged}, timestep=1), zt, xt, yt)

        tracks = tracker.get_tracks()

        # A should win (same age, bigger size)
        assert tracks[1][-1].cloud_id == 3
        assert tracks[1][-1].merges_count == 1
        assert 2 in tracks[1][-1].merged_with

        # B is the loser
        assert not tracks[2][-1].is_active
        assert tracks[2][-1].merged_into == 1
