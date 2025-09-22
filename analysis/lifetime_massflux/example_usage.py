"""
End-to-end example on synthetic clouds: fit c(z), predict, compute simple metrics.
Run: `python example_usage.py`
"""

import numpy as np
import xarray as xr
from features import reduce_per_cloud
from cz_estimator import fit_cz
from predict import predict_j
from metrics import mape_per_z

def _make_synth(nz=12, ny=12, nx=12, nt=8, dx=25.0, dy=25.0, dt=60.0, seed=0):
    rng = np.random.default_rng(seed)
    z = np.arange(nz); y = np.arange(ny); x = np.arange(nx); t = np.arange(nt)
    rho0 = 1.18 - 0.001 * z
    clouds = []
    for k in range(20):
        mask = rng.random((nz,ny,nx,nt)) > (0.85 - 0.1 * rng.random())
        w = (rng.standard_normal((nz,ny,nx,nt)) * 0.6) + 0.25 + 0.15 * (z[:,None,None,None]/nz)
        ds = xr.Dataset(
            dict(mask=(("z","y","x","t"), mask),
                 w=(("z","y","x","t"), w),
                 rho0=(("z",), rho0)),
            coords=dict(z=z, y=y, x=x, t=t),
            attrs=dict(dx=dx, dy=dy, dt=dt),
        )
        clouds.append(ds)
    return clouds

if __name__ == "__main__":
    clouds = _make_synth()
    reduced = [reduce_per_cloud(ds) for ds in clouds]
    rho0 = clouds[0]["rho0"]

    # Fit c(z) on first 15, evaluate on last 5
    train = reduced[:15]; test = reduced[15:]
    c_z = fit_cz(train, rho0, smooth=True)
    # Stack test metrics
    J_list = []; Jhat_list = []
    for ds in test:
        J_list.append(ds["J"].expand_dims(cloud=[len(J_list)]))
        Jhat_list.append(predict_j(ds, c_z, rho0).expand_dims(cloud=[len(Jhat_list)]))
    J_stack = xr.concat(J_list, dim="cloud"); Jhat_stack = xr.concat(Jhat_list, dim="cloud")

    mape = mape_per_z(J_stack, Jhat_stack).median(dim="cloud", skipna=True)
    print("Median MAPE per z (%):")
    print(np.array2string(mape.values, precision=1, separator=", "))

