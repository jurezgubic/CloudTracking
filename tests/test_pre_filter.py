"""
Tests for CloudTracker.pre_filter_cloud_matches.

Verifies centroid-based pre-filtering, which reduces the number of
expensive is_match() KD-tree calls by discarding cloud pairs whose centroids
are too far apart to possibly match:
  - Nearby / distant cloud proximity.
  - Periodic boundary wrapping (parametrized over x, y, corner).
  - Inactive track exclusion.
  - Fallback mode and disabled pre-filtering.
"""

import numpy as np
import pytest

from lib.cloud import Cloud
from lib.cloudtracker import CloudTracker
from tests.conftest import MockCloudField


def _make_prefilter_cloud(cloud_id, x, y, z, is_active=True, timestep=0):
    """Create a minimal Cloud for pre-filter testing."""
    surface_points = np.array(
        [
            [x, y, z],
            [x + 10, y, z],
            [x, y + 10, z],
        ],
        dtype=np.float32,
    )

    n_levels = 5
    return Cloud(
        cloud_id=cloud_id,
        size=10,
        surface_area=3,
        cloud_base_area=3,
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
        ql_flux=0.0,
        mass_flux=0.0,
        mass_flux_per_level=np.zeros(n_levels),
        temp_per_level=np.zeros(n_levels),
        theta_outside_per_level=np.zeros(n_levels),
        w_per_level=np.zeros(n_levels),
        circum_per_level=np.zeros(n_levels),
        eff_radius_per_level=np.zeros(n_levels),
        is_active=is_active,
    )


@pytest.fixture
def prefilter_tracker(tracker_config):
    """CloudTracker with a 10 km x 10 km domain and pre-filtering enabled."""
    config = {**tracker_config, "use_pre_filtering": True}
    tracker = CloudTracker(config)
    tracker.xt = np.linspace(0, 10000, 401)
    tracker.yt = np.linspace(0, 10000, 401)
    tracker.zt = np.linspace(0, 3000, 121)
    tracker.domain_size_x = 10000.0
    tracker.domain_size_y = 10000.0
    return tracker


# --- Basic proximity ---


class TestPreFilterProximity:
    """Basic proximity and distance checks."""

    def test_nearby_cloud_is_candidate(self, prefilter_tracker):
        prev = _make_prefilter_cloud("prev", 5000, 5000, 1000, timestep=0)
        prefilter_tracker.cloud_tracks["prev"] = [prev]

        curr = _make_prefilter_cloud("curr", 5100, 5100, 1000, timestep=1)
        field = MockCloudField({"curr": curr}, timestep=1)

        matches = prefilter_tracker.pre_filter_cloud_matches(field)
        assert "curr" in matches["prev"]

    def test_distant_cloud_excluded(self, prefilter_tracker):
        prev = _make_prefilter_cloud("prev", 1000, 1000, 1000, timestep=0)
        prefilter_tracker.cloud_tracks["prev"] = [prev]

        curr = _make_prefilter_cloud("curr", 6000, 1000, 1000, timestep=1)
        field = MockCloudField({"curr": curr}, timestep=1)

        prefilter_tracker.config["switch_prefilter_fallback"] = False
        matches = prefilter_tracker.pre_filter_cloud_matches(field)

        assert "prev" in matches
        assert len(matches["prev"]) == 0

    def test_vertical_distance_exclusion(self, prefilter_tracker):
        prev = _make_prefilter_cloud("prev", 5000, 5000, 500, timestep=0)
        prefilter_tracker.cloud_tracks["prev"] = [prev]

        curr = _make_prefilter_cloud("curr", 5000, 5000, 3500, timestep=1)
        field = MockCloudField({"curr": curr}, timestep=1)

        prefilter_tracker.config["switch_prefilter_fallback"] = False
        matches = prefilter_tracker.pre_filter_cloud_matches(field)

        assert len(matches["prev"]) == 0


# --- Periodic boundary tests (parametrized) ---


class TestPreFilterPeriodicBoundary:
    """Periodic boundary wrapping should detect close-by clouds."""

    @pytest.mark.parametrize(
        "prev_x, prev_y, curr_x, curr_y",
        [
            (9900, 5000, 100, 5000),  # x-boundary wrap
            (5000, 9800, 5000, 200),  # y-boundary wrap
            (9900, 9900, 100, 100),  # corner wrap
        ],
        ids=["x_boundary", "y_boundary", "corner"],
    )
    def test_periodic_boundary_match(self, prefilter_tracker, prev_x, prev_y, curr_x, curr_y):
        prev = _make_prefilter_cloud("prev", prev_x, prev_y, 1000, timestep=0)
        prefilter_tracker.cloud_tracks["prev"] = [prev]

        curr = _make_prefilter_cloud("curr", curr_x, curr_y, 1000, timestep=1)
        field = MockCloudField({"curr": curr}, timestep=1)

        prefilter_tracker.config["switch_prefilter_fallback"] = False
        matches = prefilter_tracker.pre_filter_cloud_matches(field)

        assert "curr" in matches["prev"]


# --- Inactive / fallback / disabled ---


class TestPreFilterEdgeCases:
    """Inactive tracks, fallback, disabled pre-filtering, empty inputs."""

    def test_inactive_track_excluded(self, prefilter_tracker):
        inactive = _make_prefilter_cloud("old", 5000, 5000, 1000, is_active=False, timestep=0)
        prefilter_tracker.cloud_tracks["old"] = [inactive]

        curr = _make_prefilter_cloud("curr", 5000, 5000, 1000, timestep=1)
        field = MockCloudField({"curr": curr}, timestep=1)

        matches = prefilter_tracker.pre_filter_cloud_matches(field)
        assert "old" not in matches

    def test_fallback_returns_all_when_no_centroid_match(self, prefilter_tracker):
        prev = _make_prefilter_cloud("prev", 1000, 1000, 1000, timestep=0)
        prefilter_tracker.cloud_tracks["prev"] = [prev]

        curr = _make_prefilter_cloud("curr", 8000, 8000, 1000, timestep=1)
        field = MockCloudField({"curr": curr}, timestep=1)

        prefilter_tracker.config["switch_prefilter_fallback"] = True
        matches = prefilter_tracker.pre_filter_cloud_matches(field)

        assert "curr" in matches["prev"]

    def test_fallback_off_returns_empty(self, prefilter_tracker):
        prev = _make_prefilter_cloud("prev", 1000, 1000, 1000, timestep=0)
        prefilter_tracker.cloud_tracks["prev"] = [prev]

        curr = _make_prefilter_cloud("curr", 8000, 8000, 1000, timestep=1)
        field = MockCloudField({"curr": curr}, timestep=1)

        prefilter_tracker.config["switch_prefilter_fallback"] = False
        matches = prefilter_tracker.pre_filter_cloud_matches(field)

        assert len(matches["prev"]) == 0

    def test_disabled_returns_all_combinations(self, prefilter_tracker):
        prefilter_tracker.config["use_pre_filtering"] = False

        prev = _make_prefilter_cloud("prev", 1000, 1000, 1000, timestep=0)
        prefilter_tracker.cloud_tracks["prev"] = [prev]

        c1 = _make_prefilter_cloud("c1", 9000, 9000, 2000, timestep=1)
        c2 = _make_prefilter_cloud("c2", 5000, 5000, 1000, timestep=1)
        field = MockCloudField({"c1": c1, "c2": c2}, timestep=1)

        matches = prefilter_tracker.pre_filter_cloud_matches(field)
        assert set(matches["prev"]) == {"c1", "c2"}

    def test_no_tracks_returns_empty(self, prefilter_tracker):
        curr = _make_prefilter_cloud("curr", 5000, 5000, 1000, timestep=1)
        field = MockCloudField({"curr": curr}, timestep=1)

        matches = prefilter_tracker.pre_filter_cloud_matches(field)
        assert len(matches) == 0

    def test_no_current_clouds_returns_empty(self, prefilter_tracker):
        prev = _make_prefilter_cloud("prev", 5000, 5000, 1000, timestep=0)
        prefilter_tracker.cloud_tracks["prev"] = [prev]

        field = MockCloudField({}, timestep=1)
        matches = prefilter_tracker.pre_filter_cloud_matches(field)

        assert len(matches) == 0
