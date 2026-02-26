"""Tests for the MONC data adapter.

Covers the three most complex physics code paths:
- _compute_reference_profiles(): hydrostatic pressure integration
- _interpolate_w_to_scalar_levels(): staggered grid interpolation
- load_timestep(): full conversion pipeline (theta reconstruction,
  theta_l conversion, total water, transposition)
"""

import sys
import os
import pytest
import numpy as np
from pathlib import Path
from netCDF4 import Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapters.monc_adapter import MONCAdapter, R_D, C_PD, L_V, G


# --- Grid constants for synthetic data ---
NX, NY = 4, 4
ZN = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0])   # scalar levels
Z_W = np.array([0.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0])  # w levels
NZ = len(ZN)
NZ_W = len(Z_W)

# Config values
DX, DY = 200.0, 200.0
P_SURFACE = 101325.0
P0 = 100000.0
Z_THREF = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0])
F_THREF = np.array([300.0, 301.0, 302.0, 303.5, 305.0])

# Synthetic field values (MONC dimension order: x, y, z)
TH_PERT = 1.0       # K
P_PERT = 50.0        # Pa
U_VAL = 2.0          # m/s
V_VAL = -1.0         # m/s
Q_VAPOUR = 0.015     # kg/kg
Q_CLOUD_LIQUID = 0.001
Q_RAIN = 0.0005
Q_ICE = 0.0
Q_SNOW = 0.0
Q_GRAUPEL = 0.0


def _write_mcf(path, z_thref=Z_THREF, f_thref=F_THREF):
    """Write a synthetic MCF config file."""
    z_str = ",".join(f"{z:.1f}" for z in z_thref)
    f_str = ",".join(f"{f:.2f}" for f in f_thref)
    content = f"""
l_thoff=.true.
dxx={DX}
dyy={DY}
surface_pressure={P_SURFACE}
surface_reference_pressure={P0}
z_init_pl_theta={z_str}
f_init_pl_theta={f_str}
"""
    path.write_text(content)


def _write_netcdf(path, q_cloud_liquid=Q_CLOUD_LIQUID,
                  th_pert_field=None, p_pert_field=None,
                  w_field=None):
    """Write a synthetic MONC NetCDF file.

    All 3-D fields have MONC dimension order (time, x, y, z/zn).
    """
    ds = Dataset(str(path), 'w', format='NETCDF4')

    ds.createDimension('time_series_0', 1)
    ds.createDimension('x', NX)
    ds.createDimension('y', NY)
    ds.createDimension('z', NZ_W)
    ds.createDimension('zn', NZ)

    # Coordinate variables
    z_var = ds.createVariable('z', 'f8', ('z',))
    z_var[:] = Z_W
    zn_var = ds.createVariable('zn', 'f8', ('zn',))
    zn_var[:] = ZN

    def _add_3d(name, fill, dim_z='zn'):
        """Add a 3-D field with shape (1, NX, NY, len(dim_z))."""
        nz_dim = NZ if dim_z == 'zn' else NZ_W
        var = ds.createVariable(name, 'f8', ('time_series_0', 'x', 'y', dim_z))
        if isinstance(fill, np.ndarray):
            var[0, :, :, :] = fill
        else:
            var[0, :, :, :] = np.full((NX, NY, nz_dim), fill)
        return var

    if th_pert_field is not None:
        _add_3d('th', th_pert_field)
    else:
        _add_3d('th', TH_PERT)

    if p_pert_field is not None:
        _add_3d('p', p_pert_field)
    else:
        _add_3d('p', P_PERT)

    _add_3d('u', U_VAL)
    _add_3d('v', V_VAL)

    # w on staggered z-levels: linear ramp by default
    if w_field is not None:
        _add_3d('w', w_field, dim_z='z')
    else:
        w_ramp = np.zeros((NX, NY, NZ_W))
        for k in range(NZ_W):
            w_ramp[:, :, k] = Z_W[k] / Z_W[-1] * 5.0  # 0..5 m/s
        _add_3d('w', w_ramp, dim_z='z')

    _add_3d('q_vapour', Q_VAPOUR)
    _add_3d('q_cloud_liquid_mass', q_cloud_liquid)
    _add_3d('q_rain_mass', Q_RAIN)
    _add_3d('q_ice_mass', Q_ICE)
    _add_3d('q_snow_mass', Q_SNOW)
    _add_3d('q_graupel_mass', Q_GRAUPEL)

    ds.close()


def _make_adapter(tmp_path, **netcdf_kwargs):
    """Build a MONCAdapter from synthetic files in tmp_path."""
    mcf_path = tmp_path / "test.mcf"
    _write_mcf(mcf_path)

    nc_path = tmp_path / "3dfields_ts_100.nc"
    _write_netcdf(nc_path, **netcdf_kwargs)

    config = {
        'monc_data_path': str(tmp_path),
        'monc_config_file': str(mcf_path),
    }
    return MONCAdapter(config)


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #

@pytest.fixture
def adapter(tmp_path):
    """A MONCAdapter backed by synthetic data (default field values)."""
    return _make_adapter(tmp_path)


@pytest.fixture
def adapter_no_liquid(tmp_path):
    """Adapter where q_cloud_liquid = 0 everywhere."""
    return _make_adapter(tmp_path, q_cloud_liquid=0.0)


# ------------------------------------------------------------------ #
#  TestComputeReferenceProfiles                                       #
# ------------------------------------------------------------------ #

class TestComputeReferenceProfiles:
    """Tests for _compute_reference_profiles (hydrostatic integration)."""

    def test_pressure_decreases_with_height(self, adapter):
        adapter.get_grid_info()
        pref = adapter._pref_on_grid
        for i in range(1, len(pref)):
            assert pref[i] < pref[i - 1], (
                f"pref should decrease with height: "
                f"pref[{i-1}]={pref[i-1]:.1f}, pref[{i}]={pref[i]:.1f}"
            )

    def test_surface_pressure_matches_input(self, adapter):
        adapter.get_grid_info()
        pref = adapter._pref_on_grid
        # First grid level is at z=0, should match surface pressure
        assert pref[0] == pytest.approx(P_SURFACE, rel=1e-10)

    def test_thref_interpolation(self, adapter):
        adapter.get_grid_info()
        expected = np.interp(ZN, Z_THREF, F_THREF)
        np.testing.assert_array_almost_equal(
            adapter._thref_on_grid, expected, decimal=10
        )

    def test_pressure_in_physical_range(self, adapter):
        adapter.get_grid_info()
        pref = adapter._pref_on_grid
        assert np.all(pref > 20_000), "Pressure should be > 20 kPa in troposphere"
        assert np.all(pref < 110_000), "Pressure should be < 110 kPa"

    def test_isothermal_atmosphere(self, tmp_path):
        """Constant theta: verify against independent hydrostatic calculation."""
        theta_const = 300.0
        z_cfg = np.array([0.0, 5000.0])
        f_cfg = np.array([theta_const, theta_const])

        mcf_path = tmp_path / "test.mcf"
        _write_mcf(mcf_path, z_thref=z_cfg, f_thref=f_cfg)
        nc_path = tmp_path / "3dfields_ts_100.nc"
        _write_netcdf(nc_path)

        config = {
            'monc_data_path': str(tmp_path),
            'monc_config_file': str(mcf_path),
        }
        adp = MONCAdapter(config)
        adp.get_grid_info()

        pref = adp._pref_on_grid
        thref = adp._thref_on_grid

        # Independent integration with the same algorithm
        expected_p = np.zeros_like(ZN)
        expected_p[0] = P_SURFACE
        for i in range(1, len(ZN)):
            dz = ZN[i] - ZN[i - 1]
            theta_avg = 0.5 * (thref[i] + thref[i - 1])
            T_avg = theta_avg * (expected_p[i - 1] / P0) ** (R_D / C_PD)
            rho_avg = expected_p[i - 1] / (R_D * T_avg)
            expected_p[i] = expected_p[i - 1] - rho_avg * G * dz

        np.testing.assert_array_almost_equal(pref, expected_p, decimal=4)


# ------------------------------------------------------------------ #
#  TestInterpolateW                                                   #
# ------------------------------------------------------------------ #

class TestInterpolateW:
    """Tests for _interpolate_w_to_scalar_levels (staggered grid)."""

    def test_linear_profile_exact(self, adapter):
        """Linear w(z) should be interpolated exactly."""
        slope = 2.0
        w_on_z = np.zeros((NX, NY, NZ_W))
        for k in range(NZ_W):
            w_on_z[:, :, k] = slope * Z_W[k]

        result = adapter._interpolate_w_to_scalar_levels(w_on_z, Z_W, ZN)

        for k in range(NZ):
            expected = slope * ZN[k]
            np.testing.assert_almost_equal(
                result[:, :, k], expected, decimal=10,
                err_msg=f"Linear interpolation inexact at zn={ZN[k]}"
            )

    def test_constant_field_preserved(self, adapter):
        w_const = 3.14
        w_on_z = np.full((NX, NY, NZ_W), w_const)
        result = adapter._interpolate_w_to_scalar_levels(w_on_z, Z_W, ZN)
        np.testing.assert_almost_equal(result, w_const, decimal=10)

    def test_output_shape(self, adapter):
        w_on_z = np.ones((NX, NY, NZ_W))
        result = adapter._interpolate_w_to_scalar_levels(w_on_z, Z_W, ZN)
        assert result.shape == (NX, NY, NZ)

    def test_boundary_below(self, adapter):
        """Scalar level below all w-levels clamps to first w value."""
        z_w_high = Z_W + 100.0  # shift w-levels up so zn[0]=0 is below
        w_on_z = np.zeros((NX, NY, NZ_W))
        w_on_z[:, :, 0] = 7.0
        w_on_z[:, :, 1:] = 1.0

        result = adapter._interpolate_w_to_scalar_levels(w_on_z, z_w_high, ZN)
        np.testing.assert_almost_equal(
            result[:, :, 0], 7.0, decimal=10,
            err_msg="Should clamp to first w value below w-grid"
        )

    def test_boundary_above(self, adapter):
        """Scalar level above all w-levels clamps to last w value."""
        z_w_low = Z_W - 1000.0  # shift w-levels down so zn[-1] is above
        z_w_low = np.maximum(z_w_low, 0)  # keep non-negative
        w_on_z = np.zeros((NX, NY, NZ_W))
        w_on_z[:, :, -1] = 9.0
        w_on_z[:, :, :-1] = 1.0

        result = adapter._interpolate_w_to_scalar_levels(w_on_z, z_w_low, ZN)
        np.testing.assert_almost_equal(
            result[:, :, -1], 9.0, decimal=10,
            err_msg="Should clamp to last w value above w-grid"
        )


# ------------------------------------------------------------------ #
#  TestLoadTimestep                                                   #
# ------------------------------------------------------------------ #

class TestLoadTimestep:
    """Tests for load_timestep — full conversion pipeline."""

    def test_output_has_all_required_fields(self, adapter):
        data = adapter.load_timestep(0)
        required = ['l', 'u', 'v', 'w', 'p', 'theta_l', 'q_t', 'r',
                     'xt', 'yt', 'zt']
        for key in required:
            assert key in data, f"Missing required field: {key}"

    def test_output_shape_is_zyx(self, adapter):
        data = adapter.load_timestep(0)
        expected_shape = (NZ, NY, NX)
        for key in ['l', 'u', 'v', 'w', 'p', 'theta_l', 'q_t', 'r']:
            assert data[key].shape == expected_shape, (
                f"{key} shape {data[key].shape} != expected {expected_shape}"
            )

    def test_coordinate_arrays(self, adapter):
        data = adapter.load_timestep(0)
        assert len(data['xt']) == NX
        assert len(data['yt']) == NY
        assert len(data['zt']) == NZ

    def test_theta_l_equals_theta_when_no_liquid(self, adapter_no_liquid):
        """With q_cloud_liquid=0, theta_l must equal reconstructed theta."""
        data = adapter_no_liquid.load_timestep(0)
        # theta = thref + th_pert; theta_l should be same when q_l=0
        thref = adapter_no_liquid._thref_on_grid
        expected_theta = thref + TH_PERT  # 1-D broadcast handled by transpose
        # After transpose: data['theta_l'] has shape (nz, ny, nx)
        for k in range(NZ):
            np.testing.assert_almost_equal(
                data['theta_l'][k, :, :], expected_theta[k], decimal=6,
                err_msg=f"theta_l != theta at level {k} when q_l=0"
            )

    def test_theta_l_conversion(self, adapter):
        """Verify theta_l = theta - (L_v/c_pd) * (q_l / Pi)."""
        data = adapter.load_timestep(0)
        thref = adapter._thref_on_grid
        pref = adapter._pref_on_grid

        for k in range(NZ):
            theta_k = thref[k] + TH_PERT
            p_k = pref[k] + P_PERT
            Pi_k = (p_k / P0) ** (R_D / C_PD)
            expected_theta_l = theta_k - (L_V / C_PD) * (Q_CLOUD_LIQUID / Pi_k)
            np.testing.assert_almost_equal(
                data['theta_l'][k, 0, 0], expected_theta_l, decimal=4,
                err_msg=f"theta_l incorrect at level {k}"
            )

    def test_total_water_composition(self, adapter):
        """q_t = q_vapour + q_cloud + q_ice + q_snow + q_graupel (no rain)."""
        data = adapter.load_timestep(0)
        expected_qt = Q_VAPOUR + Q_CLOUD_LIQUID + Q_ICE + Q_SNOW + Q_GRAUPEL
        np.testing.assert_almost_equal(
            data['q_t'], expected_qt, decimal=10,
            err_msg="q_t should equal sum of vapour + cloud + ice species"
        )

    def test_rain_excluded_from_total_water(self, adapter):
        """Rain is stored in 'r', not added to q_t."""
        data = adapter.load_timestep(0)
        np.testing.assert_almost_equal(
            data['r'], Q_RAIN, decimal=10,
            err_msg="Rain field should contain q_rain values"
        )
        # q_t should NOT include rain
        expected_qt_with_rain = Q_VAPOUR + Q_CLOUD_LIQUID + Q_RAIN
        assert not np.allclose(data['q_t'], expected_qt_with_rain), (
            "q_t should not include rain"
        )

    def test_pressure_reconstruction(self, adapter):
        """p = pref + p_pert, verified against known perturbation."""
        data = adapter.load_timestep(0)
        pref = adapter._pref_on_grid
        for k in range(NZ):
            expected_p = pref[k] + P_PERT
            np.testing.assert_almost_equal(
                data['p'][k, 0, 0], expected_p, decimal=4,
                err_msg=f"Pressure incorrect at level {k}"
            )

    def test_transposition_correctness(self, tmp_path):
        """A marker at MONC (x=0,y=1,z=2) should appear at output (z=2,y=1,x=0)."""
        # Create a NetCDF with a unique marker in th_pert
        th_field = np.full((NX, NY, NZ), TH_PERT)
        th_field[0, 1, 2] = 99.0  # marker at x=0, y=1, z=2

        adp = _make_adapter(tmp_path, th_pert_field=th_field)
        data = adp.load_timestep(0)

        # After transpose (z,y,x), the marker in theta_l should be at [2,1,0]
        thref = adp._thref_on_grid
        pref = adp._pref_on_grid
        p_marker = pref[2] + P_PERT
        Pi_marker = (p_marker / P0) ** (R_D / C_PD)
        expected_marker = (thref[2] + 99.0) - (L_V / C_PD) * (Q_CLOUD_LIQUID / Pi_marker)

        actual = data['theta_l'][2, 1, 0]
        assert actual == pytest.approx(expected_marker, rel=1e-6), (
            f"Transposition error: theta_l[2,1,0]={actual}, expected {expected_marker}"
        )

        # Non-marker point should differ
        p_other = pref[0] + P_PERT
        Pi_other = (p_other / P0) ** (R_D / C_PD)
        non_marker = (thref[0] + TH_PERT) - (L_V / C_PD) * (Q_CLOUD_LIQUID / Pi_other)
        actual_other = data['theta_l'][0, 0, 0]
        assert actual_other == pytest.approx(non_marker, rel=1e-6)


# ------------------------------------------------------------------ #
#  TestThetaLConversion                                               #
# ------------------------------------------------------------------ #

class TestThetaLConversion:
    """Focused physics validation of the theta_l conversion."""

    def test_theta_l_less_than_theta_when_cloudy(self, adapter):
        """Condensation cooling: theta_l < theta when q_l > 0."""
        data = adapter.load_timestep(0)
        thref = adapter._thref_on_grid
        for k in range(NZ):
            theta_k = thref[k] + TH_PERT
            assert np.all(data['theta_l'][k, :, :] < theta_k), (
                f"theta_l should be less than theta at level {k}"
            )

    def test_theta_l_magnitude(self, adapter):
        """For 1 g/kg liquid at ~1 bar, cooling is roughly L_v*q_l/c_pd ~ 2.5 K."""
        data = adapter.load_timestep(0)
        thref = adapter._thref_on_grid

        # Check at lowest level where pressure is closest to p0
        theta_0 = thref[0] + TH_PERT
        theta_l_0 = data['theta_l'][0, 0, 0]
        cooling = theta_0 - theta_l_0

        # Exact: (L_V / C_PD) * (q_l / Pi), with Pi ~ 1 at surface
        rough_cooling = L_V / C_PD * Q_CLOUD_LIQUID  # ~2.49 K
        assert cooling == pytest.approx(rough_cooling, rel=0.05), (
            f"Cooling {cooling:.3f} K should be ~{rough_cooling:.3f} K "
            f"for {Q_CLOUD_LIQUID*1000:.1f} g/kg liquid at surface"
        )
