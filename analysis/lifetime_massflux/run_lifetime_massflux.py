"""
Run lifetime mass-flux analysis on cloud_results.nc and print metrics.

Usage (from repo root or analysis folder):
  python analysis/lifetime_massflux/run_lifetime_massflux.py \
         --nc cloud_results.nc --dt 60 --rho0 ones --min_timesteps 3

Notes
- dt [s] is the timestep duration used when writing cloud_results.nc.
- rho0: choose a reference density profile source:
  * ones:   use rho0(z) = 1 everywhere (default). c(z) then absorbs density.
  * env:    use env_rho_mean_per_level[level] if present in the file.
  * var:    use rho0_per_level[level] if present (custom variable).
"""

from __future__ import annotations
import argparse
import numpy as np
import xarray as xr

from features import reduce_all_tracks
from cz_estimator import fit_cz
from predict import predict_j
from metrics import mape_per_z


def _load_rho0(ds: xr.Dataset, mode: str) -> xr.DataArray:
    if 'height' not in ds:
        raise ValueError("cloud_results.nc missing 'height' variable for vertical coordinate")
    z = xr.DataArray(ds['height'].values, dims=('z',), name='z')
    n = z.sizes['z']
    mode = mode.lower()
    if mode == 'ones':
        return xr.DataArray(np.ones(n, dtype=float), dims=('z',), coords=dict(z=z))
    if mode == 'env':
        if 'env_rho_mean_per_level' in ds:
            vals = np.asarray(ds['env_rho_mean_per_level'].values, dtype=float)
            if vals.size != n:
                raise ValueError("env_rho_mean_per_level length does not match number of levels")
            return xr.DataArray(vals, dims=('z',), coords=dict(z=z))
        print("Warning: env_rho_mean_per_level not found. Falling back to ones.")
        return xr.DataArray(np.ones(n, dtype=float), dims=('z',), coords=dict(z=z))
    if mode == 'var':
        name = 'rho0_per_level'
        if name in ds:
            vals = np.asarray(ds[name].values, dtype=float)
            if vals.size != n:
                raise ValueError(f"{name} length does not match number of levels")
            return xr.DataArray(vals, dims=('z',), coords=dict(z=z))
        raise ValueError("rho0_per_level not found in dataset")
    raise ValueError("--rho0 must be one of: ones, env, var")


def main(args: argparse.Namespace) -> None:
    print(f"[run] opening {args.nc} ...", flush=True)
    ds = xr.open_dataset(args.nc)
    # show some basic sizes so user knows it's alive
    print(f"[run] dims: track={ds.sizes.get('track',0)}, time={ds.sizes.get('time',0)}, level={ds.sizes.get('level',0)}", flush=True)
    rho0 = _load_rho0(ds, args.rho0)
    # Reduce all valid tracks
    print(f"[run] reducing tracks (dt={args.dt}s, rho0='{args.rho0}') ...", flush=True)
    red = reduce_all_tracks(ds, dt=float(args.dt), rho0=rho0,
                            only_valid=not args.include_partial,
                            min_timesteps=args.min_timesteps,
                            positive_only=True)
    if len(red) == 0:
        print("No clouds passed the selection. Check min_timesteps or valid_track flags.")
        return

    # Train/test split: first N-args.test_last for training, last for testing
    n = len(red)
    n_test = min(args.test_last, n//5 if n >= 5 else 1)
    n_train = max(1, n - n_test)
    train = red[:n_train]
    test = red[n_train:]
    print(f"[run] clouds: total={n}, train={len(train)}, test={len(test)}", flush=True)

    # Fit c(z)
    print("[run] fitting c(z) ...", flush=True)
    c_z = fit_cz(train, rho0=rho0, smooth=not args.no_smooth)

    # Evaluate on test
    if len(test) == 0:
        print("No test clouds; skipping metrics.")
    else:
        print("[run] predicting and scoring on test clouds ...", flush=True)
        J_list = []; Jhat_list = []
        for i, dsr in enumerate(test):
            J = dsr['J'] if 'J' in dsr else dsr['J_rho']
            J_list.append(J.expand_dims(cloud=[i]))
            Jhat_list.append(predict_j(dsr, c_z, rho0).expand_dims(cloud=[i]))
        J_stack = xr.concat(J_list, dim='cloud')
        Jhat_stack = xr.concat(Jhat_list, dim='cloud')
        mape = mape_per_z(J_stack, Jhat_stack).median(dim='cloud', skipna=True)
        overall = float(np.nanmedian(mape.values))
        print("Median MAPE per z (%):")
        print(np.array2string(mape.values, precision=1, separator=", "))
        print(f"Overall median MAPE: {overall:.2f}%")

    # Save c(z)
    out_path = args.out
    print(f"[run] saving c(z) to {out_path} ...", flush=True)
    c_z.to_dataset(name='c').to_netcdf(out_path)
    print(f"[run] done. bye.", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Lifetime mass-flux analysis on cloud_results.nc')
    p.add_argument('--nc', default='../../cloud_results.nc', help='Path to cloud_results.nc')
    p.add_argument('--dt', type=float, default=60.0, help='Timestep duration in seconds')
    p.add_argument('--rho0', default='ones', choices=['ones','env','var'], help='Reference density source')
    p.add_argument('--min_timesteps', type=int, default=3, help='Min active timesteps per track')
    p.add_argument('--test_last', type=int, default=5, help='Number of clouds for test set (last)')
    p.add_argument('--no_smooth', action='store_true', help='Disable z-smoothing for c(z)')
    p.add_argument('--include_partial', action='store_true', help='Include partial (invalid) tracks')
    p.add_argument('--out', default='analysis/lifetime_massflux/cz_estimate.nc', help='Output NetCDF for c(z)')
    main(p.parse_args())
