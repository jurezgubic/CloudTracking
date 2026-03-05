"""
Tests for CloudField.identify_regions, find_boundary_merges, and update_labels_for_merges.

These tests verify the cloud identification pipeline:
  - Liquid water thresholding (l_condition) labels connected cloudy regions.
  - The w_switch gate filters out cloud points with vertical velocity below w_condition.
  - The min_size filter (applied later in create_clouds_from_labeled_array)
    removes regions smaller than the configured minimum.
  - Periodic boundary merge detection finds cloud fragments touching
    opposite domain edges and marks them for relabeling.
  - update_labels_for_merges unifies all fragments into the minimum label ID.

The tests use small synthetic 3-D fields so they run in milliseconds.
"""

import unittest

import numpy as np

from lib.cloudfield import CloudField


class _IdentifyRegionsHelper:
    """
    Mixin providing a lightweight way to call CloudField methods without
    running the full __init__ (which needs physics, KD-trees, NIP, etc.).
    """

    def make_field(self, distance_threshold=0):
        """Return a bare CloudField-like object with only the attributes
        needed by identify_regions / find_boundary_merges."""
        obj = object.__new__(CloudField)
        obj.distance_threshold = distance_threshold
        return obj


class TestIdentifyRegionsLCondition(_IdentifyRegionsHelper, unittest.TestCase):
    """Verify that the liquid-water threshold produces correct labeled regions."""

    def test_single_cloud_above_threshold(self):
        """A single contiguous blob above l_condition is labeled as one region."""
        nz, ny, nx = 5, 10, 10
        l_data = np.zeros((nz, ny, nx))
        l_data[2, 3:6, 3:6] = 1e-3  # well above threshold

        w_data = np.ones_like(l_data)

        config = {"l_condition": 5e-4, "w_switch": False, "b_switch": False}
        cf = self.make_field()
        labeled = cf.identify_regions(l_data, w_data, config)

        self.assertEqual(np.max(labeled), 1, "Should find exactly 1 region")
        self.assertTrue(np.all(labeled[2, 3:6, 3:6] > 0), "All cloudy points should be labeled")
        self.assertEqual(np.sum(labeled > 0), 9, "Only the 9 cloudy points should be labeled")

    def test_no_cloud_below_threshold(self):
        """No regions when all liquid water is below l_condition."""
        nz, ny, nx = 5, 10, 10
        l_data = np.full((nz, ny, nx), 1e-5)  # well below threshold
        w_data = np.ones_like(l_data)

        config = {"l_condition": 5e-4, "w_switch": False, "b_switch": False}
        cf = self.make_field()
        labeled = cf.identify_regions(l_data, w_data, config)

        self.assertEqual(np.max(labeled), 0, "No regions should be found")

    def test_two_separate_clouds(self):
        """Two non-touching blobs produce two distinct labels."""
        nz, ny, nx = 5, 20, 20
        l_data = np.zeros((nz, ny, nx))
        l_data[2, 2:4, 2:4] = 1e-3
        l_data[2, 15:17, 15:17] = 1e-3

        w_data = np.ones_like(l_data)
        config = {"l_condition": 5e-4, "w_switch": False, "b_switch": False}
        cf = self.make_field()
        labeled = cf.identify_regions(l_data, w_data, config)

        self.assertEqual(np.max(labeled), 2, "Should find 2 separate regions")


class TestIdentifyRegionsWSwitch(_IdentifyRegionsHelper, unittest.TestCase):
    """Verify that w_switch filters cloud points by vertical velocity."""

    def test_w_switch_removes_downdraft_cloud(self):
        """Cloud with negative w is excluded when w_switch is True."""
        nz, ny, nx = 5, 10, 10
        l_data = np.zeros((nz, ny, nx))
        l_data[2, 3:6, 3:6] = 1e-3  # cloudy
        w_data = np.full_like(l_data, -1.0)  # everywhere downward

        config = {"l_condition": 5e-4, "w_switch": True, "w_condition": 0.0, "b_switch": False}
        cf = self.make_field()
        labeled = cf.identify_regions(l_data, w_data, config)

        self.assertEqual(np.max(labeled), 0, "Downdraft cloud should be excluded by w_switch")

    def test_w_switch_keeps_updraft_cloud(self):
        """Cloud with positive w is kept when w_switch is True."""
        nz, ny, nx = 5, 10, 10
        l_data = np.zeros((nz, ny, nx))
        l_data[2, 3:6, 3:6] = 1e-3
        w_data = np.full_like(l_data, 2.0)

        config = {"l_condition": 5e-4, "w_switch": True, "w_condition": 0.0, "b_switch": False}
        cf = self.make_field()
        labeled = cf.identify_regions(l_data, w_data, config)

        self.assertEqual(np.max(labeled), 1, "Updraft cloud should be kept")

    def test_w_switch_false_keeps_downdraft(self):
        """When w_switch is False, downdraft clouds are still labeled."""
        nz, ny, nx = 5, 10, 10
        l_data = np.zeros((nz, ny, nx))
        l_data[2, 3:6, 3:6] = 1e-3
        w_data = np.full_like(l_data, -5.0)

        config = {"l_condition": 5e-4, "w_switch": False, "b_switch": False}
        cf = self.make_field()
        labeled = cf.identify_regions(l_data, w_data, config)

        self.assertEqual(np.max(labeled), 1, "Downdraft cloud should be kept when w_switch=False")

    def test_w_switch_splits_updraft_from_downdraft(self):
        """A single cloud blob split by w_switch: only the updraft part survives."""
        nz, ny, nx = 5, 10, 10
        l_data = np.zeros((nz, ny, nx))
        l_data[2, 3:8, 3:8] = 1e-3  # 5x5 cloud
        w_data = np.zeros_like(l_data)
        w_data[2, 3:5, 3:8] = 2.0  # top 2 rows: updraft
        w_data[2, 5:8, 3:8] = -1.0  # bottom 3 rows: downdraft

        config = {"l_condition": 5e-4, "w_switch": True, "w_condition": 0.0, "b_switch": False}
        cf = self.make_field()
        labeled = cf.identify_regions(l_data, w_data, config)

        # Only the updraft portion should be labeled (2*5 = 10 points)
        self.assertEqual(np.sum(labeled > 0), 10, "Only 10 updraft points should survive")

    def test_w_condition_nonzero_threshold(self):
        """A non-zero w_condition filters out weakly rising cloud points."""
        nz, ny, nx = 5, 10, 10
        l_data = np.zeros((nz, ny, nx))
        l_data[2, 3:6, 3:6] = 1e-3
        w_data = np.full_like(l_data, 0.3)  # below 0.5 threshold

        config = {"l_condition": 5e-4, "w_switch": True, "w_condition": 0.5, "b_switch": False}
        cf = self.make_field()
        labeled = cf.identify_regions(l_data, w_data, config)

        self.assertEqual(np.max(labeled), 0, "Cloud with w < w_condition should be excluded")


class TestFindBoundaryMerges(_IdentifyRegionsHelper, unittest.TestCase):
    """Test detection of cloud fragments touching opposite periodic boundaries."""

    def test_north_south_merge(self):
        """Cloud touching both north and south boundaries is detected as a merge."""
        nz, ny, nx = 3, 10, 10
        labeled = np.zeros((nz, ny, nx), dtype=int)
        # Label 1 on south boundary (y=0)
        labeled[1, 0, 4:6] = 1
        # Label 2 on north boundary (y=9)
        labeled[1, 9, 4:6] = 2

        cf = self.make_field(distance_threshold=1)
        merges = cf.find_boundary_merges(labeled)

        self.assertEqual(len(merges), 1, "Should detect 1 merge pair")
        self.assertIn((2, 1), merges, "North label 2 and south label 1 should be merged")

    def test_east_west_merge(self):
        """Cloud touching both east and west boundaries is detected as a merge."""
        nz, ny, nx = 3, 10, 10
        labeled = np.zeros((nz, ny, nx), dtype=int)
        # Label 1 on west boundary (x=0)
        labeled[1, 4:6, 0] = 1
        # Label 2 on east boundary (x=9)
        labeled[1, 4:6, 9] = 2

        cf = self.make_field(distance_threshold=1)
        merges = cf.find_boundary_merges(labeled)

        self.assertEqual(len(merges), 1, "Should detect 1 merge pair")
        self.assertIn((2, 1), merges, "East label 2 and west label 1 should be merged")

    def test_no_merge_far_apart(self):
        """Boundary fragments at different z/y positions don't merge."""
        nz, ny, nx = 5, 10, 10
        labeled = np.zeros((nz, ny, nx), dtype=int)
        # Label 1 at z=1, y=0
        labeled[1, 0, 3] = 1
        # Label 2 at z=4, y=9  (far apart in z, threshold distance=0)
        labeled[4, 9, 3] = 2

        cf = self.make_field(distance_threshold=0)
        merges = cf.find_boundary_merges(labeled)

        # They are not within threshold distance of each other
        self.assertEqual(len(merges), 0, "Distant boundary fragments should not merge")

    def test_no_merge_without_boundary_contact(self):
        """Interior clouds (not touching any boundary) produce no merges."""
        nz, ny, nx = 5, 10, 10
        labeled = np.zeros((nz, ny, nx), dtype=int)
        labeled[2, 4:6, 4:6] = 1  # interior cloud

        cf = self.make_field(distance_threshold=1)
        merges = cf.find_boundary_merges(labeled)

        self.assertEqual(len(merges), 0, "Interior cloud should not trigger merge")


class TestUpdateLabelsForMerges(_IdentifyRegionsHelper, unittest.TestCase):
    """Test that update_labels_for_merges unifies labels correctly."""

    def test_two_labels_merged_to_minimum(self):
        """Two labels are unified to the smaller label ID."""
        labeled = np.array([[[1, 0, 2], [0, 0, 0], [1, 0, 2]]])
        cf = self.make_field()
        result = cf.update_labels_for_merges(labeled, [(1, 2)])

        self.assertTrue(np.all(result[result > 0] == 1), "All nonzero should be label 1")

    def test_chain_merge_full_unification(self):
        """Chain of merges (1-2, 2-3) — Union-Find resolves the transitive
        chain so all three labels are unified to the global minimum (1)."""
        labeled = np.array([[[1, 2, 3]]])
        cf = self.make_field()
        result = cf.update_labels_for_merges(labeled, [(1, 2), (2, 3)])

        unique_labels = set(result[result > 0].flat)
        self.assertEqual(unique_labels, {1}, "All labels should be unified to 1")

    def test_empty_merges_no_change(self):
        """Empty merge list leaves labels unchanged."""
        labeled = np.array([[[1, 0, 2]]])
        expected = labeled.copy()
        cf = self.make_field()
        result = cf.update_labels_for_merges(labeled, [])

        np.testing.assert_array_equal(result, expected)

    def test_self_merge_no_change(self):
        """Merging a label with itself leaves the array unchanged."""
        labeled = np.array([[[1, 0, 1]]])
        expected = labeled.copy()
        cf = self.make_field()
        result = cf.update_labels_for_merges(labeled, [(1, 1)])

        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
