"""
Tests for KD-tree based cloud matching.

Verifies that CloudTracker.is_match() correctly identifies cloud matches
between timesteps using surface point overlap, including drift prediction
and edge cases (inactive cloud, empty surface points).
"""

import numpy as np

from lib.cloud import Cloud
from lib.cloudtracker import CloudTracker
from tests.conftest import MockCloudField


def _make_match_cloud(cloud_id, location, surface_points, mean_u=0.0, mean_v=0.0, mean_w=0.0, is_active=True):
    """Create a Cloud tailored for is_match tests (scalar velocities)."""
    if not isinstance(surface_points, np.ndarray):
        surface_points = np.array(surface_points, dtype=np.float32)

    return Cloud(
        cloud_id=cloud_id,
        size=100,
        surface_area=50,
        cloud_base_area=20,
        cloud_base_height=location[2],
        location=location,
        points=[location],
        surface_points=surface_points,
        timestep=0,
        max_height=location[2],
        max_w=1.0,
        max_w_cloud_base=0.5,
        mean_u=mean_u,
        mean_v=mean_v,
        mean_w=mean_w,
        ql_flux=0.1,
        mass_flux=0.2,
        mass_flux_per_level=[0.2],
        temp_per_level=[290],
        theta_outside_per_level=[300],
        w_per_level=[1.0],
        circum_per_level=[50],
        eff_radius_per_level=[20],
        is_active=is_active,
    )


class TestCloudMatching:
    """Test cloud matching using KD-tree based surface point overlap."""

    def _tracker(self, tracker_config):
        """Build a CloudTracker with a 1 km domain."""
        tracker = CloudTracker(tracker_config)
        tracker.zt = np.array([100, 300, 500, 700, 900])
        tracker.xt = np.linspace(0, 1000, 41)
        tracker.yt = np.linspace(0, 1000, 41)
        tracker.domain_size_x = 1000.0
        tracker.domain_size_y = 1000.0
        return tracker

    def test_exact_match(self, tracker_config):
        """Clouds with identical surface points match."""
        tr = self._tracker(tracker_config)
        pts = [(100, 100, 500), (125, 100, 500), (100, 125, 500)]

        cloud_t0 = _make_match_cloud(1, (100, 100, 500), pts)
        cloud_t1 = _make_match_cloud(2, (100, 100, 500), pts)
        cf = MockCloudField({cloud_t1.cloud_id: cloud_t1})

        assert tr.is_match(cloud_t1, cloud_t0, cf)

    def test_nearby_match(self, tracker_config):
        """Slightly offset clouds within threshold distance match."""
        tr = self._tracker(tracker_config)

        cloud_t0 = _make_match_cloud(1, (100, 100, 500), [(100, 100, 500)])
        cloud_t1 = _make_match_cloud(2, (110, 110, 500), [(110, 110, 500)])
        cf = MockCloudField({cloud_t1.cloud_id: cloud_t1})

        assert tr.is_match(cloud_t1, cloud_t0, cf)

    def test_no_match_distance(self, tracker_config):
        """Clouds beyond the threshold distance do not match."""
        tr = self._tracker(tracker_config)

        cloud_t0 = _make_match_cloud(1, (100, 100, 500), [(100, 100, 500)])
        cloud_t1 = _make_match_cloud(2, (150, 100, 500), [(150, 100, 500)])
        cf = MockCloudField({cloud_t1.cloud_id: cloud_t1})

        assert not tr.is_match(cloud_t1, cloud_t0, cf)

    def test_drift_prediction_match(self, tracker_config):
        """Cloud velocity is used to predict position for matching."""
        tr = self._tracker(tracker_config)

        cloud_t0 = _make_match_cloud(
            1, (100, 100, 500), [(100, 100, 500)], mean_u=0.4
        )  # 0.4 m/s * 60 s = 24 m displacement
        cloud_t1 = _make_match_cloud(2, (124, 100, 500), [(124, 100, 500)])
        cf = MockCloudField({cloud_t1.cloud_id: cloud_t1})

        assert tr.is_match(cloud_t1, cloud_t0, cf)

    def test_vertical_drift_match(self, tracker_config):
        """Vertical drift is correctly applied for matching."""
        tr = self._tracker(tracker_config)

        cloud_t0 = _make_match_cloud(
            1, (100, 100, 500), [(100, 100, 500)], mean_w=0.3
        )  # 0.3 m/s * 60 s = 18 m vertical
        cloud_t1 = _make_match_cloud(2, (100, 100, 518), [(100, 100, 518)])
        cf = MockCloudField({cloud_t1.cloud_id: cloud_t1})

        assert tr.is_match(cloud_t1, cloud_t0, cf)

    def test_large_vertical_movement_no_match(self, tracker_config):
        """Too much unexplained vertical movement prevents matching."""
        tr = self._tracker(tracker_config)

        cloud_t0 = _make_match_cloud(1, (100, 100, 500), [(100, 100, 500)])
        cloud_t1 = _make_match_cloud(2, (100, 100, 550), [(100, 100, 550)])
        cf = MockCloudField({cloud_t1.cloud_id: cloud_t1})

        assert not tr.is_match(cloud_t1, cloud_t0, cf)

    def test_inactive_cloud_no_match(self, tracker_config):
        """Inactive clouds are not matched."""
        tr = self._tracker(tracker_config)

        cloud_t0 = _make_match_cloud(1, (100, 100, 500), [(100, 100, 500)], is_active=False)
        cloud_t1 = _make_match_cloud(2, (100, 100, 500), [(100, 100, 500)])
        cf = MockCloudField({cloud_t1.cloud_id: cloud_t1})

        assert not tr.is_match(cloud_t1, cloud_t0, cf)

    def test_empty_surface_points_no_match(self, tracker_config):
        """Clouds with empty surface points don\'t match."""
        tr = self._tracker(tracker_config)

        cloud_t0 = _make_match_cloud(1, (100, 100, 500), np.array([], dtype=np.float32).reshape(0, 3))
        cloud_t1 = _make_match_cloud(2, (100, 100, 500), [(100, 100, 500)])
        cf = MockCloudField({cloud_t1.cloud_id: cloud_t1})

        assert not tr.is_match(cloud_t1, cloud_t0, cf)
