"""
Shared pytest fixtures for cloud tracking tests.

Provides reusable factories and domain setup used across multiple test modules:
  - tracker_config: default CloudTracker configuration dict
  - domain_grids: zt, xt, yt arrays for a small test domain
  - make_cloud: factory function for lightweight Cloud objects
  - make_tracker: ready-to-use CloudTracker with domain initialised
  - MockCloudField: mock cloud field with KD-tree infrastructure
  - SimpleMockCloudField: minimal mock without KD-tree (for merge/split tests)
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from lib.cloud import Cloud
from lib.cloudtracker import CloudTracker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker_config():
    """Default CloudTracker configuration for tests."""
    return {
        "min_size": 3,
        "timestep_duration": 60,
        "horizontal_resolution": 25.0,
        "switch_wind_drift": False,
        "switch_background_drift": False,
        "switch_vertical_drift": False,
        "max_expected_cloud_speed": 20.0,
        "bounding_box_safety_factor": 2.0,
        "use_pre_filtering": False,
        "switch_prefilter_fallback": True,
        "match_safety_factor_dynamic": 2.0,
        "min_h_match_factor": 1.0,
        "min_v_match_factor": 1.0,
        "min_surface_overlap_points": 1,
    }


# ---------------------------------------------------------------------------
# Domain grids
# ---------------------------------------------------------------------------


@pytest.fixture
def domain_grids():
    """Small test domain grids (500 m × 500 m × 6 levels)."""
    zt = np.array([0, 100, 200, 300, 400, 500])
    xt = np.arange(0, 500, 25.0)
    yt = np.arange(0, 500, 25.0)
    return zt, xt, yt


# ---------------------------------------------------------------------------
# Cloud factory
# ---------------------------------------------------------------------------


def make_cloud(
    cloud_id,
    size,
    location,
    timestep,
    n_levels,
    is_active=True,
    mean_u=None,
    mean_v=None,
    mean_w=None,
    surface_points=None,
):
    """Create a lightweight Cloud for testing.

    Parameters
    ----------
    cloud_id : hashable
        Unique cloud identifier.
    size : int
        Cloud size (cell count).
    location : tuple (x, y, z)
        Physical centroid.
    timestep : int
        Timestep index.
    n_levels : int
        Number of vertical levels (for per-level arrays).
    is_active : bool
        Whether the cloud is active.
    mean_u, mean_v, mean_w : float or array, optional
        Mean velocities.  Scalars or arrays of length *n_levels*.
    surface_points : array-like, optional
        Explicit surface points.  Defaults to a single point at *location*.
    """
    if mean_u is None:
        mean_u = np.zeros(n_levels)
    if mean_v is None:
        mean_v = np.zeros(n_levels)
    if mean_w is None:
        mean_w = np.zeros(n_levels)
    if surface_points is None:
        surface_points = np.array([location], dtype=np.float32)
    elif not isinstance(surface_points, np.ndarray):
        surface_points = np.array(surface_points, dtype=np.float32)

    return Cloud(
        cloud_id=cloud_id,
        size=size,
        surface_area=size * 2,
        cloud_base_area=max(1, size // 2),
        cloud_base_height=location[2],
        location=location,
        points=[location],
        surface_points=surface_points,
        timestep=timestep,
        max_height=location[2],
        max_w=1.0,
        max_w_cloud_base=0.5,
        mean_u=mean_u,
        mean_v=mean_v,
        mean_w=mean_w,
        ql_flux=0.1,
        mass_flux=0.2,
        mass_flux_per_level=np.zeros(n_levels),
        temp_per_level=np.zeros(n_levels),
        theta_outside_per_level=np.zeros(n_levels),
        w_per_level=np.zeros(n_levels),
        circum_per_level=np.zeros(n_levels),
        eff_radius_per_level=np.zeros(n_levels),
        is_active=is_active,
    )


# ---------------------------------------------------------------------------
# Tracker factory
# ---------------------------------------------------------------------------


@pytest.fixture
def make_tracker(tracker_config, domain_grids):
    """Factory fixture: returns a ready-to-use CloudTracker with domain set."""

    def _make(**config_overrides):
        cfg = {**tracker_config, **config_overrides}
        tracker = CloudTracker(cfg)
        zt, xt, yt = domain_grids
        tracker.zt = zt
        tracker.xt = xt
        tracker.yt = yt
        tracker.domain_size_x = float(xt[-1] - xt[0]) + cfg["horizontal_resolution"]
        tracker.domain_size_y = float(yt[-1] - yt[0]) + cfg["horizontal_resolution"]
        return tracker

    return _make


# ---------------------------------------------------------------------------
# Mock cloud fields
# ---------------------------------------------------------------------------


class MockCloudField:
    """Mock CloudField with KD-tree infrastructure for surface-point overlap tests."""

    def __init__(self, clouds_dict, timestep=0):
        self.clouds = clouds_dict
        self.timestep = timestep

        self.surface_points_array = None
        self.surface_point_to_cloud_id = None
        self.surface_points_kdtree = None
        self._build_surface_kdtree()

    def _build_surface_kdtree(self):
        if not self.clouds:
            return

        total_points = 0
        for cloud in self.clouds.values():
            if hasattr(cloud, "surface_points") and cloud.surface_points is not None:
                pts = cloud.surface_points
                if isinstance(pts, np.ndarray):
                    total_points += pts.shape[0]
                else:
                    total_points += len(pts)

        if total_points == 0:
            return

        self.surface_points_array = np.empty((total_points, 3), dtype=np.float32)
        self.surface_point_to_cloud_id = np.empty(total_points, dtype=object)

        idx = 0
        for cloud_id, cloud in self.clouds.items():
            if hasattr(cloud, "surface_points") and cloud.surface_points is not None:
                points = cloud.surface_points
                if not isinstance(points, np.ndarray):
                    points = np.array(points)
                if points.ndim == 1:
                    points = points.reshape(1, -1)
                n = points.shape[0]
                if n > 0:
                    self.surface_points_array[idx : idx + n] = points
                    self.surface_point_to_cloud_id[idx : idx + n] = cloud_id
                    idx += n

        if idx < total_points:
            self.surface_points_array = self.surface_points_array[:idx]
            self.surface_point_to_cloud_id = self.surface_point_to_cloud_id[:idx]

        if len(self.surface_points_array) > 0:
            self.surface_points_kdtree = cKDTree(self.surface_points_array[:, :2])


class SimpleMockCloudField:
    """Minimal cloud field mock (no KD-tree).  Used by merge/split tests that
    monkey-patch ``is_match``."""

    def __init__(self, clouds_dict, timestep=0):
        self.clouds = clouds_dict
        self.timestep = timestep
