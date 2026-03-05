"""
Tests for cloud splitting behaviour.

Verifies:
  1. Track Continuation: The LARGEST fragment continues the parent track (by cell count).
  2. Split Inheritance: Other fragments start new tracks with age = parent + 1.
  3. Split Provenance: The continuation child does NOT carry split_from (it IS the parent).
  4. New Cloud Handling: Genuinely new clouds start with age 0.
"""

import numpy as np
from lib.cloudtracker import CloudTracker
from tests.conftest import make_cloud, SimpleMockCloudField


class TestCloudSplitting:
    """Test the cloud splitting behavior in CloudTracker."""

    def test_largest_child_continues_parent_track(
            self, tracker_config, domain_grids):
        """The largest child (by cell count) continues the parent track.

        T0: Cloud P (id=1, size=10)
        T1: Cloud big (id=2, size=6), Cloud small (id=3, size=4) both match P.
            Cloud new (id=4, size=8) matches nothing.

        Expected:
          - Track 1 = [P, big] (big is largest child, continues)
          - Track 3 = [small]  (new track, split_from=1)
          - Track 4 = [new]    (genuinely new, age=0)
        """
        zt, xt, yt = domain_grids
        n = len(zt)
        tracker = CloudTracker(tracker_config)

        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                return last_cloud.cloud_id == 1 and cloud.cloud_id in [2, 3]
            return False

        tracker.is_match = mock_is_match

        # T0
        cloud_p = make_cloud(1, 10, (100, 100, 200), 0, n)
        tracker.update_tracks(
            SimpleMockCloudField({1: cloud_p}, timestep=0),
            zt, xt, yt)

        # T1
        cloud_big = make_cloud(2, 6, (102, 102, 210), 1, n)
        cloud_small = make_cloud(3, 4, (97, 99, 205), 1, n)
        cloud_new = make_cloud(4, 8, (300, 300, 250), 1, n)
        tracker.update_tracks(
            SimpleMockCloudField(
                {2: cloud_big, 3: cloud_small, 4: cloud_new}, timestep=1),
            zt, xt, yt)

        tracks = tracker.get_tracks()

        # Parent track continues with the LARGER child (cloud_big, size=6)
        parent_track = tracks[1]
        assert len(parent_track) == 2
        assert parent_track[0].cloud_id == 1
        assert parent_track[1].cloud_id == 2, "Largest child should continue"
        assert parent_track[1].age == 1

        # Continuation child should NOT have split_from (it IS the parent track)
        assert parent_track[1].split_from is None,             "Child continuing parent must not carry split_from"

        # Smaller child starts a new track with provenance
        split_track = tracks[3]
        assert len(split_track) == 1
        assert split_track[0].cloud_id == 3
        assert split_track[0].age == 1
        assert split_track[0].split_from == 1,             "Split child should reference parent track"

        # New cloud starts fresh
        new_track = tracks[4]
        assert len(new_track) == 1
        assert new_track[0].age == 0
        assert new_track[0].split_from is None
