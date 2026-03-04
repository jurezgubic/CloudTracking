"""
Tests for CloudTracker.pre_filter_cloud_matches.

These tests verify centroid-based pre-filtering, which reduces the number of
expensive is_match() KD-tree calls by discarding cloud pairs whose centroids
are too far apart to possibly match:
  - Nearby clouds (within search_radius) are returned as candidates.
  - Distant clouds (beyond search_radius) are excluded.
  - Periodic boundary wrapping: clouds near opposite domain edges are
    correctly identified as nearby.
  - Inactive tracks are excluded from candidate generation.
  - The fallback mode (switch_prefilter_fallback) returns all clouds
    when the centroid filter finds no candidates.
  - When pre-filtering is disabled entirely, all combinations are returned.

All tests build lightweight CloudTracker + mock cloud objects to avoid
needing real LES data.
"""

import unittest
import numpy as np
from lib.cloudtracker import CloudTracker
from lib.cloud import Cloud
from tests.test_utils import MockCloudField


def _make_config(**overrides):
    """Return a test config with pre-filtering enabled by default."""
    config = {
        'horizontal_resolution': 25.0,
        'timestep_duration': 60,
        'max_expected_cloud_speed': 20.0,  # search_radius = 20*60*2 = 2400 m
        'bounding_box_safety_factor': 2.0,
        'use_pre_filtering': True,
        'switch_prefilter_fallback': True,
        'switch_background_drift': False,
        'switch_wind_drift': False,
        'switch_vertical_drift': False,
        'match_safety_factor_dynamic': 2.0,
        'min_h_match_factor': 1.0,
        'min_v_match_factor': 1.0,
        'min_surface_overlap_points': 1,
    }
    config.update(overrides)
    return config


def _make_cloud(cloud_id, x, y, z, is_active=True, timestep=0):
    """Create a minimal Cloud for pre-filter testing."""
    surface_points = np.array([
        [x, y, z], [x + 10, y, z], [x, y + 10, z],
    ], dtype=np.float32)

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


class TestPreFilterCloudMatches(unittest.TestCase):
    """Test centroid-based pre-filtering with periodic boundaries."""

    def setUp(self):
        """Set up a tracker with a 10 km × 10 km domain."""
        self.config = _make_config()
        self.tracker = CloudTracker(self.config)

        # 10 km domain
        self.tracker.xt = np.linspace(0, 10000, 401)
        self.tracker.yt = np.linspace(0, 10000, 401)
        self.tracker.zt = np.linspace(0, 3000, 121)
        self.tracker.domain_size_x = 10000.0
        self.tracker.domain_size_y = 10000.0

    # --- Basic proximity tests ---

    def test_nearby_cloud_is_candidate(self):
        """A cloud within the search radius is returned as a candidate."""
        # Previous cloud at (5000, 5000, 1000)
        prev_cloud = _make_cloud("prev", 5000, 5000, 1000, timestep=0)
        self.tracker.cloud_tracks["prev"] = [prev_cloud]

        # Current cloud at (5100, 5100, 1000) — 141 m away, radius is 2400 m
        curr_cloud = _make_cloud("curr", 5100, 5100, 1000, timestep=1)
        field = MockCloudField({"curr": curr_cloud}, timestep=1)

        matches = self.tracker.pre_filter_cloud_matches(field)

        self.assertIn("prev", matches)
        self.assertIn("curr", matches["prev"])

    def test_distant_cloud_excluded(self):
        """A cloud far beyond the search radius is not a candidate."""
        # search_radius = 20 m/s * 60 s * 2.0 = 2400 m
        prev_cloud = _make_cloud("prev", 1000, 1000, 1000, timestep=0)
        self.tracker.cloud_tracks["prev"] = [prev_cloud]

        # Current cloud 5000 m away (> 2400 m)
        curr_cloud = _make_cloud("curr", 6000, 1000, 1000, timestep=1)
        field = MockCloudField({"curr": curr_cloud}, timestep=1)

        matches = self.tracker.pre_filter_cloud_matches(field)

        # With fallback ON, distant cloud still appears (fallback kicks in)
        # Disable fallback to test pure proximity filtering
        self.tracker.config['switch_prefilter_fallback'] = False
        matches = self.tracker.pre_filter_cloud_matches(field)

        self.assertIn("prev", matches)
        self.assertEqual(len(matches["prev"]), 0,
                         "Distant cloud should not be a candidate with fallback off")

    def test_vertical_distance_exclusion(self):
        """Clouds within horizontal range but too far apart vertically are excluded."""
        prev_cloud = _make_cloud("prev", 5000, 5000, 500, timestep=0)
        self.tracker.cloud_tracks["prev"] = [prev_cloud]

        # Same x,y but z differs by 3000 m (> search_radius of 2400 m)
        curr_cloud = _make_cloud("curr", 5000, 5000, 3500, timestep=1)
        field = MockCloudField({"curr": curr_cloud}, timestep=1)

        self.tracker.config['switch_prefilter_fallback'] = False
        matches = self.tracker.pre_filter_cloud_matches(field)

        self.assertEqual(len(matches["prev"]), 0,
                         "Vertically distant cloud should be excluded")

    # --- Periodic boundary tests ---

    def test_periodic_x_boundary_match(self):
        """Cloud near x=0 matches cloud near x=domain_size (periodic wrap)."""
        # Previous cloud near east edge
        prev_cloud = _make_cloud("prev", 9900, 5000, 1000, timestep=0)
        self.tracker.cloud_tracks["prev"] = [prev_cloud]

        # Current cloud near west edge — periodic distance ~ 200 m
        curr_cloud = _make_cloud("curr", 100, 5000, 1000, timestep=1)
        field = MockCloudField({"curr": curr_cloud}, timestep=1)

        self.tracker.config['switch_prefilter_fallback'] = False
        matches = self.tracker.pre_filter_cloud_matches(field)

        self.assertIn("curr", matches["prev"],
                      "Periodic x-boundary match should be detected")

    def test_periodic_y_boundary_match(self):
        """Cloud near y=0 matches cloud near y=domain_size (periodic wrap)."""
        prev_cloud = _make_cloud("prev", 5000, 9800, 1000, timestep=0)
        self.tracker.cloud_tracks["prev"] = [prev_cloud]

        curr_cloud = _make_cloud("curr", 5000, 200, 1000, timestep=1)
        field = MockCloudField({"curr": curr_cloud}, timestep=1)

        self.tracker.config['switch_prefilter_fallback'] = False
        matches = self.tracker.pre_filter_cloud_matches(field)

        self.assertIn("curr", matches["prev"],
                      "Periodic y-boundary match should be detected")

    def test_periodic_corner_wrap(self):
        """Cloud at domain corner (9900,9900) matches cloud at (100,100)."""
        prev_cloud = _make_cloud("prev", 9900, 9900, 1000, timestep=0)
        self.tracker.cloud_tracks["prev"] = [prev_cloud]

        curr_cloud = _make_cloud("curr", 100, 100, 1000, timestep=1)
        field = MockCloudField({"curr": curr_cloud}, timestep=1)

        self.tracker.config['switch_prefilter_fallback'] = False
        matches = self.tracker.pre_filter_cloud_matches(field)

        # Periodic distance = sqrt(200^2 + 200^2) ~ 283 m < 2400 m
        self.assertIn("curr", matches["prev"],
                      "Corner periodic wrap should be detected")

    # --- Inactive track handling ---

    def test_inactive_track_excluded(self):
        """Inactive tracks produce no candidates."""
        inactive_cloud = _make_cloud("old", 5000, 5000, 1000,
                                     is_active=False, timestep=0)
        self.tracker.cloud_tracks["old"] = [inactive_cloud]

        curr_cloud = _make_cloud("curr", 5000, 5000, 1000, timestep=1)
        field = MockCloudField({"curr": curr_cloud}, timestep=1)

        matches = self.tracker.pre_filter_cloud_matches(field)

        self.assertNotIn("old", matches,
                         "Inactive track should not appear in matches")

    # --- Fallback behaviour ---

    def test_fallback_returns_all_when_no_centroid_match(self):
        """With fallback ON, all clouds returned when centroid filter finds nothing."""
        prev_cloud = _make_cloud("prev", 1000, 1000, 1000, timestep=0)
        self.tracker.cloud_tracks["prev"] = [prev_cloud]

        # Far-away cloud — centroid filter would exclude it
        curr_cloud = _make_cloud("curr", 8000, 8000, 1000, timestep=1)
        field = MockCloudField({"curr": curr_cloud}, timestep=1)

        self.tracker.config['switch_prefilter_fallback'] = True
        matches = self.tracker.pre_filter_cloud_matches(field)

        self.assertIn("curr", matches["prev"],
                      "Fallback should include the distant cloud")

    def test_fallback_off_returns_empty(self):
        """With fallback OFF, distant cloud yields empty candidate list."""
        prev_cloud = _make_cloud("prev", 1000, 1000, 1000, timestep=0)
        self.tracker.cloud_tracks["prev"] = [prev_cloud]

        curr_cloud = _make_cloud("curr", 8000, 8000, 1000, timestep=1)
        field = MockCloudField({"curr": curr_cloud}, timestep=1)

        self.tracker.config['switch_prefilter_fallback'] = False
        matches = self.tracker.pre_filter_cloud_matches(field)

        self.assertEqual(len(matches["prev"]), 0,
                         "No fallback should give empty candidates for distant cloud")

    # --- Disabled pre-filtering ---

    def test_disabled_returns_all_combinations(self):
        """When use_pre_filtering=False, all current clouds are candidates for every track."""
        self.tracker.config['use_pre_filtering'] = False

        prev_cloud = _make_cloud("prev", 1000, 1000, 1000, timestep=0)
        self.tracker.cloud_tracks["prev"] = [prev_cloud]

        c1 = _make_cloud("c1", 9000, 9000, 2000, timestep=1)
        c2 = _make_cloud("c2", 5000, 5000, 1000, timestep=1)
        field = MockCloudField({"c1": c1, "c2": c2}, timestep=1)

        matches = self.tracker.pre_filter_cloud_matches(field)

        self.assertEqual(set(matches["prev"]), {"c1", "c2"},
                         "Disabled pre-filtering should return all clouds")

    # --- Empty inputs ---

    def test_no_tracks_returns_empty(self):
        """No existing tracks produces empty match dict."""
        curr_cloud = _make_cloud("curr", 5000, 5000, 1000, timestep=1)
        field = MockCloudField({"curr": curr_cloud}, timestep=1)

        matches = self.tracker.pre_filter_cloud_matches(field)
        self.assertEqual(len(matches), 0)

    def test_no_current_clouds_returns_empty(self):
        """No current clouds produces empty candidate lists."""
        prev_cloud = _make_cloud("prev", 5000, 5000, 1000, timestep=0)
        self.tracker.cloud_tracks["prev"] = [prev_cloud]

        field = MockCloudField({}, timestep=1)
        matches = self.tracker.pre_filter_cloud_matches(field)

        self.assertEqual(len(matches), 0)


if __name__ == '__main__':
    unittest.main()
