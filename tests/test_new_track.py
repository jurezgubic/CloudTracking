"""
Tests for cloud track creation.

Verifies that CloudTracker correctly creates new tracks, extends existing
tracks when matched, and handles inactive clouds properly.
"""

import numpy as np
from lib.cloudtracker import CloudTracker
from lib.cloud import Cloud
from tests.conftest import MockCloudField


def _make_track_cloud(cloud_id, x, y, z, is_active=True, age=0, timestep=0):
    """Create a Cloud with a 5-point surface (for real KD-tree matching)."""
    surface_points = np.array([
        [x, y, z],
        [x + 10, y, z],
        [x, y + 10, z],
        [x - 10, y, z],
        [x, y - 10, z],
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
        timestep=timestep,
        max_height=z,
        max_w=1.0,
        max_w_cloud_base=0.5,
        mean_u=0.0,
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
        age=age,
    )


class TestCloudTracker:
    """Test track creation and continuation via real KD-tree matching."""

    def _tracker(self, tracker_config):
        """Build a CloudTracker with a 1 km domain."""
        tracker = CloudTracker(tracker_config)
        tracker.zt = np.linspace(0, 1000, 10)
        tracker.xt = np.linspace(0, 1000, 41)
        tracker.yt = np.linspace(0, 1000, 41)
        tracker.domain_size_x = 1000.0
        tracker.domain_size_y = 1000.0
        return tracker

    def test_first_cloud_creates_new_track(self, tracker_config):
        """The first cloud creates a new track."""
        tracker = self._tracker(tracker_config)

        cloud = _make_track_cloud(1, 100, 100, 500)
        field = MockCloudField({cloud.cloud_id: cloud})
        tracker.update_tracks(field, tracker.zt, tracker.xt, tracker.yt)

        assert len(tracker.cloud_tracks) == 1

    def test_matched_cloud_extends_track(self, tracker_config):
        """A matched cloud extends an existing track."""
        tracker = self._tracker(tracker_config)

        cloud_t1 = _make_track_cloud(1, 100, 100, 500, timestep=0)
        field_t1 = MockCloudField({cloud_t1.cloud_id: cloud_t1}, timestep=0)
        tracker.update_tracks(field_t1, tracker.zt, tracker.xt, tracker.yt)

        cloud_t2 = _make_track_cloud(2, 100, 100, 500, timestep=1)
        field_t2 = MockCloudField({cloud_t2.cloud_id: cloud_t2}, timestep=1)
        tracker.update_tracks(field_t2, tracker.zt, tracker.xt, tracker.yt)

        assert len(tracker.cloud_tracks) == 1
        track = list(tracker.cloud_tracks.values())[0]
        assert len(track) == 2

    def test_inactive_cloud_doesnt_match_new_cloud(self, tracker_config):
        """New clouds are not matched to inactive tracks.

        T0: Cloud A appears.
        T1: A continues (matched, age 1).
        T2: A marked inactive.  Cloud B appears far away -> new track.
        """
        tracker = self._tracker(tracker_config)

        cloud_a_t1 = _make_track_cloud(1, 100, 100, 500, timestep=0)
        field_t1 = MockCloudField({cloud_a_t1.cloud_id: cloud_a_t1}, timestep=0)
        tracker.update_tracks(field_t1, tracker.zt, tracker.xt, tracker.yt)

        cloud_a_t2 = _make_track_cloud(2, 100, 100, 500, age=1, timestep=1)
        field_t2 = MockCloudField({cloud_a_t2.cloud_id: cloud_a_t2}, timestep=1)
        tracker.update_tracks(field_t2, tracker.zt, tracker.xt, tracker.yt)

        track_ids = list(tracker.cloud_tracks.keys())
        assert len(track_ids) == 1
        track_a = tracker.cloud_tracks[track_ids[0]]
        assert len(track_a) == 2

        # Mark inactive
        track_a[-1].is_active = False

        cloud_b_t3 = _make_track_cloud(3, 500, 500, 400, timestep=2)
        field_t3 = MockCloudField({cloud_b_t3.cloud_id: cloud_b_t3}, timestep=2)
        tracker.update_tracks(field_t3, tracker.zt, tracker.xt, tracker.yt)

        assert len(tracker.cloud_tracks) == 2
        assert len(tracker.cloud_tracks[track_ids[0]]) == 2

        track_b_id = [t for t in tracker.cloud_tracks if t != track_ids[0]][0]
        track_b = tracker.cloud_tracks[track_b_id]
        assert len(track_b) == 1
        assert track_b[0].cloud_id == 3
        assert track_b[0].age == 0
