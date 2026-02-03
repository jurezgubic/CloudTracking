"""
Test rain water handling in physics calculations.

Rain water (r) should be:
1. Excluded from saturation adjustment (doesn't participate in condensation)
2. Included in density loading (increases air density)

These tests verify that the physics functions properly handle non-zero rain water.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.physics import (
    calculate_physics_variables,
    _saturation_adjustment_newton,
    _density_with_loading
)


class TestRainPhysics(unittest.TestCase):
    """Test rain water handling in physics calculations."""

    def setUp(self):
        """Set up common test values."""
        # Typical RICO conditions
        self.p = 95000.0  # Pa (cloud layer pressure)
        self.theta_l = 298.0  # K (typical theta_l)
        self.horizontal_resolution_squared = 625.0  # 25m^2

    def test_rain_excluded_from_saturation(self):
        """
        Test that rain is excluded from saturation adjustment.
        
        The saturation adjustment should use r_condensable = q_t - r_r,
        so adding rain should NOT change the diagnosed temperature or
        liquid water content (only the density loading).
        """
        # Set up saturated air parcel
        q_v = np.array([0.015])  # 15 g/kg vapor
        q_l = np.array([0.001])  # 1 g/kg cloud liquid
        p_values = np.array([self.p])
        theta_l_values = np.array([self.theta_l])
        w_values = np.array([1.0])
        
        # Calculate physics WITHOUT rain
        temps_no_rain, rhos_no_rain, _ = calculate_physics_variables(
            p_values, theta_l_values, q_l, q_v, w_values,
            self.horizontal_resolution_squared,
            r_rain_values=None
        )
        
        # Calculate physics WITH zero rain (should be identical to no rain)
        temps_zero_rain, rhos_zero_rain, _ = calculate_physics_variables(
            p_values, theta_l_values, q_l, q_v, w_values,
            self.horizontal_resolution_squared,
            r_rain_values=np.array([0.0])
        )
        
        # Verify zero rain gives same result as no rain
        np.testing.assert_almost_equal(
            temps_zero_rain[0], temps_no_rain[0], decimal=6,
            err_msg="Zero rain should give same temperature as no rain"
        )
        np.testing.assert_almost_equal(
            rhos_zero_rain[0], rhos_no_rain[0], decimal=6,
            err_msg="Zero rain should give same density as no rain"
        )
        
        # Calculate physics WITH significant rain
        r_rain = np.array([0.002])  # 2 g/kg rain
        temps_with_rain, rhos_with_rain, _ = calculate_physics_variables(
            p_values, theta_l_values, q_l, q_v, w_values,
            self.horizontal_resolution_squared,
            r_rain_values=r_rain
        )
        
        # Temperature should be nearly identical (rain doesn't affect saturation)
        # Small differences may occur due to the reduced r_condensable
        # but the effect should be minimal
        temp_diff = abs(temps_with_rain[0] - temps_no_rain[0])
        self.assertLess(
            temp_diff, 0.5,  # Less than 0.5 K difference
            f"Rain should have minimal effect on temperature, got {temp_diff} K difference"
        )
        
        # Density should be HIGHER with rain (rain adds to loading)
        self.assertGreater(
            rhos_with_rain[0], rhos_no_rain[0],
            "Density should increase with rain water loading"
        )

    def test_rain_increases_density_loading(self):
        """
        Test that rain increases density through water loading.
        
        The density formula is:
            T_v = T * (1 + r_v/epsilon) / (1 + r_total)
            rho = p / (R_d * T_v)
        
        where r_total = r_v + r_l + r_r
        
        Adding rain increases r_total, decreases T_v, and increases rho.
        """
        T = 290.0  # K
        r_v = 0.015  # 15 g/kg vapor
        r_l = 0.001  # 1 g/kg liquid
        
        # Density without rain
        rho_no_rain = _density_with_loading(T, self.p, r_v, r_l, r_r=0.0)
        
        # Density with rain
        r_r = 0.002  # 2 g/kg rain
        rho_with_rain = _density_with_loading(T, self.p, r_v, r_l, r_r=r_r)
        
        # Rain should increase density
        self.assertGreater(
            rho_with_rain, rho_no_rain,
            "Rain should increase density through water loading"
        )
        
        # Calculate expected density increase
        # r_total increases by r_r, so T_v decreases, so rho increases
        r_total_no_rain = r_v + r_l
        r_total_with_rain = r_v + r_l + r_r
        
        # Fractional increase should be roughly proportional to r_r
        # (for small r_r compared to r_total)
        density_increase = (rho_with_rain - rho_no_rain) / rho_no_rain
        expected_min_increase = r_r / (1 + r_total_no_rain) * 0.5  # Rough lower bound
        
        self.assertGreater(
            density_increase, expected_min_increase,
            f"Density increase ({density_increase:.6f}) should be proportional to rain amount"
        )

    def test_heavy_rain_loading(self):
        """
        Test density loading with heavy rain (typical tropical convection).
        
        In heavy tropical rain, rain mixing ratios can reach 5-10 g/kg.
        This should significantly increase air density.
        """
        T = 285.0  # K (cooler, higher in cloud)
        r_v = 0.012  # 12 g/kg vapor
        r_l = 0.002  # 2 g/kg liquid
        
        rain_levels = [0.0, 0.001, 0.005, 0.010]  # 0, 1, 5, 10 g/kg
        densities = []
        
        for r_r in rain_levels:
            rho = _density_with_loading(T, self.p, r_v, r_l, r_r=r_r)
            densities.append(rho)
        
        # Verify monotonic increase in density with rain
        for i in range(1, len(densities)):
            self.assertGreater(
                densities[i], densities[i-1],
                f"Density should increase monotonically with rain: "
                f"r_r={rain_levels[i-1]} gives {densities[i-1]:.6f}, "
                f"r_r={rain_levels[i]} gives {densities[i]:.6f}"
            )
        
        # Heavy rain (10 g/kg) should increase density by roughly 1%
        # compared to no rain
        heavy_rain_increase = (densities[-1] - densities[0]) / densities[0]
        self.assertGreater(
            heavy_rain_increase, 0.005,  # At least 0.5% increase
            f"Heavy rain should significantly increase density, got {heavy_rain_increase*100:.2f}%"
        )

    def test_saturation_adjustment_with_reduced_condensable(self):
        """
        Test that saturation adjustment correctly handles reduced condensable water
        when rain is present.
        
        If rain takes some of the total water, there's less available for
        condensation, which could affect the diagnosed liquid water content.
        
        Physics: Less condensable water means less liquid can condense, which
        means less latent heat release, which means lower temperature.
        The temperature difference can be significant (~3-4 K for 5 g/kg rain).
        """
        theta_l = 298.0  # K
        p = 90000.0  # Pa
        
        # High total water content
        r_total = 0.020  # 20 g/kg total
        
        # Case 1: All water is condensable (no rain)
        r_condensable_1 = r_total
        T1, r_l_1, r_v_1 = _saturation_adjustment_newton(theta_l, p, r_condensable_1)
        
        # Case 2: Some water is rain (less condensable)
        r_rain = 0.005  # 5 g/kg rain
        r_condensable_2 = r_total - r_rain
        T2, r_l_2, r_v_2 = _saturation_adjustment_newton(theta_l, p, r_condensable_2)
        
        # With less condensable water, there should be less diagnosed liquid
        # (assuming air is saturated in both cases)
        if r_l_1 > 0 and r_l_2 > 0:  # Both cases saturated
            self.assertLess(
                r_l_2, r_l_1,
                f"Less condensable water should mean less diagnosed liquid: "
                f"r_l={r_l_1:.6f} (no rain) vs r_l={r_l_2:.6f} (with rain)"
            )
        
        # Temperature should be LOWER with rain because less condensate forms
        # and less latent heat is released
        self.assertLess(
            T2, T1,
            f"Temperature should be lower with rain (less latent heat release): "
            f"T={T1:.2f} K (no rain) vs T={T2:.2f} K (with rain)"
        )
        
        # The difference should scale with the amount of rain
        # Roughly: delta_T ~ L_v * delta_r_l / c_pd ~ 2.5e6 * 0.005 / 1005 ~ 12 K max
        # But actual difference is smaller due to nonlinear saturation curve
        temp_diff = T1 - T2
        self.assertGreater(
            temp_diff, 0.5,  # At least 0.5 K difference
            f"Temperature difference should be significant: {temp_diff:.2f} K"
        )
        self.assertLess(
            temp_diff, 15.0,  # No more than 15 K difference
            f"Temperature difference should be reasonable: {temp_diff:.2f} K"
        )

    def test_vectorized_rain_handling(self):
        """
        Test that rain is handled correctly in vectorized physics calculations.
        """
        n_points = 100
        
        # Create arrays of test values
        p_values = np.full(n_points, self.p)
        theta_l_values = np.full(n_points, self.theta_l)
        q_l_values = np.full(n_points, 0.001)  # 1 g/kg liquid
        q_v_values = np.full(n_points, 0.015)  # 15 g/kg vapor
        w_values = np.full(n_points, 1.0)  # 1 m/s updraft
        
        # Varying rain amounts (0 to 5 g/kg)
        r_rain_values = np.linspace(0, 0.005, n_points)
        
        # Calculate physics
        temps, rhos, mass_fluxes = calculate_physics_variables(
            p_values, theta_l_values, q_l_values, q_v_values, w_values,
            self.horizontal_resolution_squared,
            r_rain_values=r_rain_values
        )
        
        # Verify outputs are valid
        self.assertEqual(len(temps), n_points)
        self.assertEqual(len(rhos), n_points)
        self.assertEqual(len(mass_fluxes), n_points)
        
        # Temperatures should be in a reasonable range
        self.assertTrue(np.all(temps > 270), "All temperatures should be > 270 K")
        self.assertTrue(np.all(temps < 310), "All temperatures should be < 310 K")
        
        # Densities should increase monotonically with rain
        # (small variations due to saturation adjustment are acceptable)
        rho_trend = np.polyfit(r_rain_values, rhos, 1)[0]  # Linear fit slope
        self.assertGreater(
            rho_trend, 0,
            "Density should generally increase with rain amount"
        )


if __name__ == '__main__':
    unittest.main()
