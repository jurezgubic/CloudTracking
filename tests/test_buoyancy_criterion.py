import unittest
import numpy as np
from utils.physics import compute_buoyancy_3d, P_0, R_D, C_PD, L_V, EPSILON, G
from lib.cloudfield import CloudField


class TestComputeBuoyancy3D(unittest.TestCase):
    """Test the vectorized buoyancy computation."""

    def test_uniform_field_zero_buoyancy(self):
        """A horizontally uniform field should have zero buoyancy everywhere."""
        nz, ny, nx = 5, 4, 4
        theta_l = np.full((nz, ny, nx), 300.0)
        q_t = np.full((nz, ny, nx), 0.01)
        q_l = np.zeros((nz, ny, nx))
        p = np.full((nz, ny, nx), 90000.0)

        buoyancy = compute_buoyancy_3d(theta_l, q_t, q_l, p)

        np.testing.assert_allclose(buoyancy, 0.0, atol=1e-12)

    def test_warm_anomaly_positive_buoyancy(self):
        """A warm column should have positive buoyancy relative to the domain mean."""
        nz, ny, nx = 5, 10, 10
        theta_l = np.full((nz, ny, nx), 300.0)
        q_t = np.full((nz, ny, nx), 0.01)
        q_l = np.zeros((nz, ny, nx))
        p = np.full((nz, ny, nx), 90000.0)

        # Warm column at (5,5)
        theta_l[:, 5, 5] = 302.0

        buoyancy = compute_buoyancy_3d(theta_l, q_t, q_l, p)

        # Warm column should be positively buoyant
        self.assertTrue(np.all(buoyancy[:, 5, 5] > 0))
        # Background should be slightly negatively buoyant (mean pulled up by warm column)
        self.assertTrue(np.all(buoyancy[:, 0, 0] < 0))

    def test_cold_anomaly_negative_buoyancy(self):
        """A cold column should have negative buoyancy."""
        nz, ny, nx = 5, 10, 10
        theta_l = np.full((nz, ny, nx), 300.0)
        q_t = np.full((nz, ny, nx), 0.01)
        q_l = np.zeros((nz, ny, nx))
        p = np.full((nz, ny, nx), 90000.0)

        # Cold column
        theta_l[:, 5, 5] = 298.0

        buoyancy = compute_buoyancy_3d(theta_l, q_t, q_l, p)

        self.assertTrue(np.all(buoyancy[:, 5, 5] < 0))

    def test_more_vapor_increases_buoyancy(self):
        """More water vapor (lighter molecule) should increase buoyancy."""
        nz, ny, nx = 5, 10, 10
        theta_l = np.full((nz, ny, nx), 300.0)
        q_t = np.full((nz, ny, nx), 0.01)
        q_l = np.zeros((nz, ny, nx))
        p = np.full((nz, ny, nx), 90000.0)

        # Moister column: more vapor = lighter = more buoyant
        q_t[:, 5, 5] = 0.015

        buoyancy = compute_buoyancy_3d(theta_l, q_t, q_l, p)

        self.assertTrue(np.all(buoyancy[:, 5, 5] > 0))


class TestBuoyancyCriterionInIdentifyRegions(unittest.TestCase):
    """Test that the buoyancy criterion filters cloud points in identify_regions."""

    def _make_config(self, b_switch=False, b_condition=0.0):
        return {
            'distance_threshold': 100.0,
            'w_switch': False,
            'b_switch': b_switch,
            'b_condition': b_condition,
            'l_condition': 1e-4,
            'min_size': 1,
            'horizontal_resolution': 25.0,
            'plot_switch': False,
            'cloud_base_altitude': 0.0,
            'env_aloft_levels': -1,
            'env_aloft_mode': 'flat',
            'env_aloft_sampling_mode': 'exact',
            'env_ring_max_distance': 1,
            'env_periodic_rings': False,
            'match_shell_layers': 1,
            'base_scan_levels': 3,
            'base_increase_threshold': 1.5,
            'cloud_batch_size': 50,
            'nip_gamma': 0.3,
            'nip_f': 3.0,
            'nip_Lh_min': 100.0,
            'nip_Lh_max': 2000.0,
            'nip_T_min': 60.0,
            'nip_T_max': 6000.0,
        }

    def _make_data(self, nz=10, ny=20, nx=20):
        """Create baseline 3D fields: uniform environment, no clouds."""
        l = np.zeros((nz, ny, nx))
        u = np.zeros((nz, ny, nx))
        v = np.zeros((nz, ny, nx))
        w = np.zeros((nz, ny, nx))
        p = np.full((nz, ny, nx), 90000.0)
        theta_l = np.full((nz, ny, nx), 300.0)
        q_t = np.full((nz, ny, nx), 0.015)
        r = np.zeros((nz, ny, nx))
        xt = np.arange(nx) * 25.0
        yt = np.arange(ny) * 25.0
        zt = np.arange(nz) * 100.0
        return l, u, v, w, p, theta_l, q_t, r, xt, yt, zt

    def test_b_switch_false_keeps_all_clouds(self):
        """With b_switch=False, all cloudy points are kept regardless of buoyancy."""
        config = self._make_config(b_switch=False)
        l, u, v, w, p, theta_l, q_t, r, xt, yt, zt = self._make_data()

        # Two cloud columns: one warm (buoyant), one cold (negatively buoyant)
        l[3:6, 5, 5] = 0.001
        l[3:6, 15, 15] = 0.001
        theta_l[:, 5, 5] = 303.0   # warm
        theta_l[:, 15, 15] = 297.0  # cold

        cf = CloudField(l, u, v, w, p, theta_l, q_t, r, 0, config, xt, yt, zt)
        # Both clouds should be present
        self.assertEqual(len(cf.clouds), 2)

    def test_b_switch_true_excludes_negatively_buoyant(self):
        """With b_switch=True and b_condition=0, negatively buoyant cloud is excluded."""
        config = self._make_config(b_switch=True, b_condition=0.0)
        l, u, v, w, p, theta_l, q_t, r, xt, yt, zt = self._make_data()

        # Buoyant cloud column
        l[3:6, 5, 5] = 0.001
        theta_l[:, 5, 5] = 305.0

        # Negatively buoyant cloud column
        l[3:6, 15, 15] = 0.001
        theta_l[:, 15, 15] = 295.0

        cf = CloudField(l, u, v, w, p, theta_l, q_t, r, 0, config, xt, yt, zt)
        # Only the buoyant cloud should survive
        self.assertEqual(len(cf.clouds), 1)

    def test_b_condition_threshold(self):
        """b_condition threshold controls minimum allowed buoyancy."""
        l, u, v, w, p, theta_l, q_t, r, xt, yt, zt = self._make_data()

        # Single slightly warm cloud
        l[3:6, 5, 5] = 0.001
        theta_l[:, 5, 5] = 300.5  # slightly warm

        # With a low threshold it passes
        config_low = self._make_config(b_switch=True, b_condition=0.0)
        cf_low = CloudField(l.copy(), u, v, w, p, theta_l.copy(), q_t, r, 0, config_low, xt, yt, zt)
        self.assertEqual(len(cf_low.clouds), 1)

        # With a high threshold it is excluded
        config_high = self._make_config(b_switch=True, b_condition=0.1)
        cf_high = CloudField(l.copy(), u, v, w, p, theta_l.copy(), q_t, r, 0, config_high, xt, yt, zt)
        self.assertEqual(len(cf_high.clouds), 0)


if __name__ == '__main__':
    unittest.main()
