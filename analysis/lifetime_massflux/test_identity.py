"""
Synthetic identity test: J(z) == rho0(z) * T_c * tilde_a(z) * tilde_w_a(z)
Run: `python test_identity.py`
"""

import numpy as np
import xarray as xr
from features import reduce_per_cloud

def _make_synth(nz=10, ny=8, nx=8, nt=6, dx=25.0, dy=25.0, dt=60.0):
    z = np.arange(nz); y = np.arange(ny); x = np.arange(nx); t = np.arange(nt)
    rng = np.random.default_rng(0)
    # Random blob mask varying in time
    mask = rng.random((nz,ny,nx,nt)) > 0.8
    # Positive/negative w
    w = (rng.standard_normal((nz,ny,nx,nt)) * 0.5) + 0.3
    rho0 = 1.2 - 0.001 * z
    ds = xr.Dataset(
        dict(mask=(("z","y","x","t"), mask),
             w=(("z","y","x","t"), w),
             rho0=(("z",), rho0)),
        coords=dict(z=z, y=y, x=x, t=t),
        attrs=dict(dx=dx, dy=dy, dt=dt),
    )
    return ds

if __name__ == "__main__":
    ds = _make_synth()
    red = reduce_per_cloud(ds)
    T_c = red.attrs["T_c"]
    lhs = red["J"]
    rhs = ds["rho0"] * T_c * red["tilde_a"] * red["tilde_w_a"]
    err = np.nanmax(np.abs(lhs.values - rhs.values))
    print("max abs error:", err)
    assert err < 1e-6, "Identity failed"
    print("OK")

