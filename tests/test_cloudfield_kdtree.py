"""
Tests for CloudField.build_global_surface_kdtree.

These tests verify that the real CloudField KD-tree builder:
  - Aggregates surface points from multiple Cloud objects into a single array.
  - Maps every surface point back to its owning cloud_id.
  - Builds a 2-D cKDTree (X, Y only) that supports spatial neighbour queries.
  - Handles edge cases: a single cloud, an empty cloud dict, and clouds
    with varying numbers of surface points.

The tests construct minimal CloudField-like objects by injecting pre-built
Cloud objects directly, bypassing the full __init__ pipeline.
"""

import unittest
import numpy as np
from lib.cloudfield import CloudField
from lib.cloud import Cloud


def _make_bare_cloudfield(clouds_dict):
    """Create a CloudField with only the attributes needed by build_global_surface_kdtree."""
    obj = object.__new__(CloudField)
    obj.clouds = clouds_dict
    obj.surface_points_array = None
    obj.surface_point_to_cloud_id = None
    obj.surface_points_kdtree = None
    return obj


def _make_cloud(cloud_id, surface_points):
    """Create a minimal Cloud with the given surface points."""
    sp = np.array(surface_points, dtype=np.float32)
    n_levels = 5
    return Cloud(
        cloud_id=cloud_id,
        size=len(sp),
        surface_area=len(sp),
        cloud_base_area=1,
        cloud_base_height=sp[0, 2] if len(sp) > 0 else 0,
        location=tuple(sp.mean(axis=0)) if len(sp) > 0 else (0, 0, 0),
        points=[(0, 0, 0)],
        surface_points=sp,
        timestep=0,
        max_height=100,
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
    )


class TestBuildGlobalSurfaceKDTree(unittest.TestCase):
    """Test the real CloudField.build_global_surface_kdtree implementation."""

    def test_single_cloud_points_and_ids(self):
        """Surface points and cloud-ID mapping are correct for a single cloud."""
        cloud = _make_cloud("0-1", [[10, 20, 30], [40, 50, 60]])
        cf = _make_bare_cloudfield({"0-1": cloud})
        cf.build_global_surface_kdtree()

        self.assertEqual(cf.surface_points_array.shape, (2, 3))
        np.testing.assert_array_almost_equal(
            cf.surface_points_array, [[10, 20, 30], [40, 50, 60]])
        self.assertTrue(np.all(cf.surface_point_to_cloud_id == "0-1"))

    def test_multiple_clouds_aggregated(self):
        """Points from two clouds are aggregated; IDs map to the correct cloud."""
        cloud_a = _make_cloud("0-1", [[0, 0, 100], [10, 0, 100]])
        cloud_b = _make_cloud("0-2", [[500, 500, 200]])
        cf = _make_bare_cloudfield({"0-1": cloud_a, "0-2": cloud_b})
        cf.build_global_surface_kdtree()

        self.assertEqual(cf.surface_points_array.shape[0], 3,
                         "Total surface points should be 2 + 1")
        # Verify cloud-ID mapping
        ids = set(cf.surface_point_to_cloud_id)
        self.assertEqual(ids, {"0-1", "0-2"})

        # Cloud A's points should map to cloud A
        mask_a = cf.surface_point_to_cloud_id == "0-1"
        self.assertEqual(np.sum(mask_a), 2)
        mask_b = cf.surface_point_to_cloud_id == "0-2"
        self.assertEqual(np.sum(mask_b), 1)

    def test_kdtree_uses_xy_only(self):
        """The KD-tree is built on X,Y only; querying returns correct indices."""
        cloud = _make_cloud("0-1", [
            [100, 200, 300],
            [100, 200, 500],  # same X,Y, different Z
            [900, 900, 300],
        ])
        cf = _make_bare_cloudfield({"0-1": cloud})
        cf.build_global_surface_kdtree()

        # Query near (100, 200) should return the first two points
        indices = cf.surface_points_kdtree.query_ball_point([100, 200], r=1.0)
        self.assertEqual(len(indices), 2,
                         "Two points at (100,200) should be returned")

    def test_kdtree_query_radius(self):
        """Querying with a radius finds only points within that distance."""
        cloud_a = _make_cloud("0-1", [[0, 0, 0], [10, 0, 0]])
        cloud_b = _make_cloud("0-2", [[1000, 1000, 0]])
        cf = _make_bare_cloudfield({"0-1": cloud_a, "0-2": cloud_b})
        cf.build_global_surface_kdtree()

        # Query radius 15 around (0,0) — should get both points of cloud_a
        indices = cf.surface_points_kdtree.query_ball_point([0, 0], r=15.0)
        matched_ids = set(cf.surface_point_to_cloud_id[indices])
        self.assertEqual(matched_ids, {"0-1"},
                         "Only cloud_a's points should be within radius")

    def test_empty_cloud_dict(self):
        """Empty cloud dict exits early; KD-tree stays None."""
        cf = _make_bare_cloudfield({})
        cf.build_global_surface_kdtree()

        self.assertIsNone(cf.surface_points_array)
        self.assertIsNone(cf.surface_points_kdtree)

    def test_many_clouds_aggregation_count(self):
        """Stress: 50 clouds with 4 points each produce 200-point tree."""
        clouds = {}
        for i in range(50):
            pts = [[i * 100 + dx, i * 100 + dy, 500]
                   for dx, dy in [(0, 0), (10, 0), (0, 10), (10, 10)]]
            clouds[f"0-{i}"] = _make_cloud(f"0-{i}", pts)
        cf = _make_bare_cloudfield(clouds)
        cf.build_global_surface_kdtree()

        self.assertEqual(cf.surface_points_array.shape[0], 200)
        self.assertEqual(len(cf.surface_point_to_cloud_id), 200)
        self.assertIsNotNone(cf.surface_points_kdtree)


if __name__ == '__main__':
    unittest.main()
