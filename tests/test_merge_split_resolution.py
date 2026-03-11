"""
Tests for merge/split resolution edge cases (Issues 1, 3, 4, 6).

Covers:
  - Stale merged_into cleared on split continuation (Issue 1)
  - Merge winner criterion: age vs size (Issue 3), parametrized
  - Downgraded merge orphan receives split_from provenance (Issue 4)
"""

import pytest

from lib.cloudtracker import CloudTracker
from tests.conftest import SimpleMockCloudField, make_cloud


class TestStaleMergedIntoClearedOnSplitContinuation:
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

    def test_merged_into_cleared_when_track_survives(self, tracker_config, domain_grids):
        zt, xt, yt = domain_grids
        n = len(zt)
        tracker = CloudTracker(tracker_config)

        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                if cloud.cloud_id == 3 and last_cloud.cloud_id in [1, 2]:
                    return True
                if cloud.cloud_id == 4 and last_cloud.cloud_id == 2:
                    return True
            return False

        tracker.is_match = mock_is_match

        # T0
        cloud_a = make_cloud(1, 10, (100, 100, 200), 0, n)
        cloud_b = make_cloud(2, 8, (200, 200, 200), 0, n)
        tracker.update_tracks(SimpleMockCloudField({1: cloud_a, 2: cloud_b}, timestep=0), zt, xt, yt)

        # T1
        cloud_m = make_cloud(3, 18, (110, 110, 210), 1, n)
        cloud_s = make_cloud(4, 6, (210, 210, 210), 1, n)
        tracker.update_tracks(SimpleMockCloudField({3: cloud_m, 4: cloud_s}, timestep=1), zt, xt, yt)

        tracks = tracker.get_tracks()

        # Track 1 won the merge: [A, M]
        assert len(tracks[1]) == 2
        assert tracks[1][-1].cloud_id == 3

        # Track 2 survived via S: [B, S]
        assert len(tracks[2]) == 2
        assert tracks[2][-1].cloud_id == 4
        assert tracks[2][-1].is_active

        # B's merged_into was set in Pass 2 but must be cleared in Pass 3
        cloud_b_in_track = tracks[2][0]
        assert cloud_b_in_track.merged_into is None, "merged_into must be cleared when track survives via split child"

        # Track 2 survived, so the stale merge record on M must also be cleaned up
        assert 2 not in tracks[1][-1].merged_with, "merged_with must drop track that survived"


class TestMergeWinnerCriterion:
    """Issue 3: Parametrized test for merge_winner_criterion (age vs size).

    T0: Cloud A (id=1, size=5)
    T1: A continues as A' (id=2, size=5), new Cloud B (id=3, size=20)
    T2: Cloud C (id=4) matched by both A' and B.
        With 'age': A' wins (age=1 > 0).
        With 'size': B wins (size=20 > 5).
    """

    @staticmethod
    def _run_merge(tracker_config, domain_grids, criterion):
        zt, xt, yt = domain_grids
        n = len(zt)
        config = {**tracker_config, "merge_winner_criterion": criterion}
        tracker = CloudTracker(config)

        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                return last_cloud.cloud_id == 1 and cloud.cloud_id == 2
            if cf.timestep == 2:
                return cloud.cloud_id == 4 and last_cloud.cloud_id in [2, 3]
            return False

        tracker.is_match = mock_is_match

        tracker.update_tracks(
            SimpleMockCloudField({1: make_cloud(1, 5, (100, 100, 200), 0, n)}, timestep=0), zt, xt, yt
        )
        tracker.update_tracks(
            SimpleMockCloudField(
                {
                    2: make_cloud(2, 5, (102, 102, 210), 1, n),
                    3: make_cloud(3, 20, (300, 300, 250), 1, n),
                },
                timestep=1,
            ),
            zt,
            xt,
            yt,
        )
        tracker.update_tracks(
            SimpleMockCloudField({4: make_cloud(4, 25, (120, 120, 220), 2, n)}, timestep=2), zt, xt, yt
        )

        return tracker.get_tracks()

    @pytest.mark.parametrize(
        "criterion, winner_track, expected_age",
        [
            ("age", 1, 2),  # A' (age=1) beats B (age=0) -> winner age=2
            ("size", 3, 1),  # B (size=20) beats A' (size=5) -> winner age=1
        ],
        ids=["age_criterion", "size_criterion"],
    )
    def test_merge_winner(self, tracker_config, domain_grids, criterion, winner_track, expected_age):
        tracks = self._run_merge(tracker_config, domain_grids, criterion)

        assert 4 in [c.cloud_id for c in tracks[winner_track]]
        assert tracks[winner_track][-1].age == expected_age

    def test_size_criterion_loser_marked(self, tracker_config, domain_grids):
        """When B wins by size, A's track is the loser."""
        tracks = self._run_merge(tracker_config, domain_grids, "size")

        assert not tracks[1][-1].is_active
        assert tracks[1][-1].merged_into == 3


class TestDowngradedMergeOrphanProvenance:
    """Issue 4: When all parents of a merge candidate are already committed,
    the orphaned cloud should still receive split_from provenance.

    T0: A (id=1, size=30), B (id=2, size=25), X (id=3, size=15), Y (id=4, size=10)
    T1: M1 (id=5) matched by A, X, Y -> 3 parents -> A wins.
        M2 (id=6) matched by B, X, Y -> 3 parents -> B wins.
        Orphan (id=7) matched by A, B -> 2 parents -> both committed -> downgraded.

    Expected: Orphan starts a new track with split_from set.
    """

    def test_orphan_gets_split_from(self, tracker_config, domain_grids):
        zt, xt, yt = domain_grids
        n = len(zt)
        tracker = CloudTracker(tracker_config)

        def mock_is_match(cloud, last_cloud, cf):
            if cf.timestep == 1:
                if cloud.cloud_id == 5 and last_cloud.cloud_id in [1, 3, 4]:
                    return True
                if cloud.cloud_id == 6 and last_cloud.cloud_id in [2, 3, 4]:
                    return True
                if cloud.cloud_id == 7 and last_cloud.cloud_id in [1, 2]:
                    return True
            return False

        tracker.is_match = mock_is_match

        # T0
        tracker.update_tracks(
            SimpleMockCloudField(
                {
                    1: make_cloud(1, 30, (100, 100, 200), 0, n),
                    2: make_cloud(2, 25, (200, 200, 200), 0, n),
                    3: make_cloud(3, 15, (300, 300, 200), 0, n),
                    4: make_cloud(4, 10, (400, 400, 200), 0, n),
                },
                timestep=0,
            ),
            zt,
            xt,
            yt,
        )

        # T1
        tracker.update_tracks(
            SimpleMockCloudField(
                {
                    5: make_cloud(5, 40, (110, 110, 210), 1, n),
                    6: make_cloud(6, 35, (210, 210, 210), 1, n),
                    7: make_cloud(7, 8, (150, 150, 210), 1, n),
                },
                timestep=1,
            ),
            zt,
            xt,
            yt,
        )

        tracks = tracker.get_tracks()

        # M1 in track 1, M2 in track 2
        assert 5 in [c.cloud_id for c in tracks[1]]
        assert 6 in [c.cloud_id for c in tracks[2]]

        # Orphan starts a new track with provenance
        orphan_track = tracks[7]
        assert len(orphan_track) == 1
        assert orphan_track[0].cloud_id == 7
        assert orphan_track[0].split_from is not None, "Orphaned merge child must have split_from provenance"
        assert orphan_track[0].split_from in [1, 2], "split_from should point to one of the committed parents"
