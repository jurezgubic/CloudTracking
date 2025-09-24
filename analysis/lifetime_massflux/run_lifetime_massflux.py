"""
Run lifetime mass-flux analysis on cloud_results.nc and print metrics.

Usage (from repo root or analysis folder):
  python analysis/lifetime_massflux/run_lifetime_massflux.py \
         --nc cloud_results.nc --dt 60 --rho0 raw --min_timesteps 3

Notes
- dt [s] is the timestep duration used when writing cloud_results.nc.
- rho0: choose a reference density profile source:
  * raw:   compute rho0(z) from raw LES files under --raw_base.
  * env:   use env_rho_mean_per_level[level] if present in the file.
  * var:   use rho0_per_level[level] if present (custom variable).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import time

from features import reduce_all_tracks
from cz_estimator import fit_cz
from predict import predict_j
from metrics import mape_per_z
from density import compute_rho0_from_raw


def _load_rho0(ds: xr.Dataset, mode: str,
               raw_base: str | None = None,
               file_map: dict | None = None,
               sample_frac: float = 1.0,
               time_max: int | None = None) -> xr.DataArray | None:
    if 'height' not in ds:
        print("[run] cloud_results.nc missing 'height' variable for vertical coordinate")
        return None
    z = xr.DataArray(ds['height'].values, dims=('z',), name='z')
    n = z.sizes['z']
    mode = mode.lower()
    if mode == 'raw':
        # Use raw LES fields to compute rho0 via thermodynamic relation
        nt = ds.sizes.get('time', None)
        if raw_base is None:
            print("[run] raw_base path is required for --rho0 raw")
            return None
        tids = np.arange(nt) if nt is not None else None
        rho0 = compute_rho0_from_raw(
            raw_base,
            file_map=file_map,
            time_indices=tids,
            reduce='median',
            sample_frac=sample_frac,
            sample_seed=0,
            time_max=time_max,
        )
        if rho0 is None:
            return None
        if rho0.sizes['z'] != n:
            print("[run] rho0(z) from raw files does not match cloud_results level count")
            return None
        return rho0
    if mode == 'env':
        if 'env_rho_mean_per_level' in ds:
            vals = np.asarray(ds['env_rho_mean_per_level'].values, dtype=float)
            if vals.size != n:
                print("[run] env_rho_mean_per_level length does not match number of levels")
                return None
            return xr.DataArray(vals, dims=('z',), coords=dict(z=z))
        print("[run] env_rho_mean_per_level not found in cloud_results.nc")
        return None
    if mode == 'var':
        name = 'rho0_per_level'
        if name in ds:
            vals = np.asarray(ds[name].values, dtype=float)
            if vals.size != n:
                print(f"[run] {name} length does not match number of levels")
                return None
            return xr.DataArray(vals, dims=('z',), coords=dict(z=z))
        print("[run] rho0_per_level not found in dataset")
        return None
    print("[run] --rho0 must be one of: raw, env, var")
    return None


def main(args: argparse.Namespace) -> None:
    t_all = time.time()
    print(f"[run] opening {args.nc} ...", flush=True)
    ds = xr.open_dataset(args.nc)
    # show some basic sizes so I know it is alive
    print(f"[run] dims: track={ds.sizes.get('track',0)}, time={ds.sizes.get('time',0)}, level={ds.sizes.get('level',0)}", flush=True)
    # Build rho0(z) from requested source
    if args.rho0.lower() == 'raw':
        print(f"[run] computing rho0(z) from raw at {args.raw_base} with sample_frac={args.rho_sample_frac} and time_max={args.rho_time_max} ... this can take a while", flush=True)
    else:
        print(f"[run] loading rho0(z) source = {args.rho0}", flush=True)
    t_rho = time.time()
    rho0 = _load_rho0(ds, args.rho0, raw_base=args.raw_base, sample_frac=float(args.rho_sample_frac or 1.0), time_max=args.rho_time_max)
    if rho0 is not None:
        print(f"[run] rho0 ready in {time.time()-t_rho:.1f}s", flush=True)
    if rho0 is None or not np.isfinite(rho0.values).any():
        print("[run] could not compute rho0(z). stopping.")
        return
    # Reduce all valid tracks
    print(f"[run] reducing tracks (dt={args.dt}s, rho0='{args.rho0}') ...", flush=True)
    t_red = time.time()
    # prefer dt from dataset attrs
    dt_eff = float(ds.attrs.get('timestep_duration_seconds', args.dt))
    red = reduce_all_tracks(ds, dt=dt_eff, rho0=rho0,
                            only_valid=not args.include_partial,
                            min_timesteps=args.min_timesteps,
                            positive_only=True)
    if len(red) == 0:
        print("No clouds passed the selection. Check min_timesteps or valid_track flags.")
        return
    print(f"[run] reduced clouds: {len(red)} in {time.time()-t_red:.1f}s", flush=True)

    # Train/test split: first N-args.test_last for training, last for testing
    n = len(red)
    n_test = min(args.test_last, n//5 if n >= 5 else 1)
    n_train = max(1, n - n_test)
    train = red[:n_train]
    test = red[n_train:]
    print(f"[run] clouds: total={n}, train={len(train)}, test={len(test)}", flush=True)

    # Fit c(z)
    print("[run] fitting c(z) ...", flush=True)
    t_fit = time.time()
    cz_ds = fit_cz(train, rho0=rho0, smooth=not args.no_smooth, use=("J_rho" if args.use_field_density else "auto"))
    c_z = cz_ds['c']
    target = cz_ds.attrs.get('target','J')
    print(f"[run] fit done in {time.time()-t_fit:.1f}s (target={target})", flush=True)

    # Evaluate on test
    if len(test) == 0:
        print("No test clouds; skipping metrics.")
    else:
        print("[run] predicting and scoring on test clouds ...", flush=True)
        t_eval = time.time()
        J_list = []; Jhat_list = []
        for i, dsr in enumerate(test):
            J = dsr[target]
            J_list.append(J.expand_dims(cloud=[i]))
            Jhat_list.append(predict_j(dsr, c_z, rho0).expand_dims(cloud=[i]))
        J_stack = xr.concat(J_list, dim='cloud')
        Jhat_stack = xr.concat(Jhat_list, dim='cloud')
        # restrict metrics to occupied levels per cloud
        Sa_stack = xr.concat([dsr['S_a'].expand_dims(cloud=[i]) for i, dsr in enumerate(test)], dim='cloud')
        present = Sa_stack > 0
        mape = mape_per_z(J_stack, Jhat_stack).where(present)
        # 2D per-cloud matrix
        print("Median MAPE per z (%), per-cloud rows:")
        print(np.array2string(mape.values, precision=1, separator=", "))
        # 1D per-z median and overall across occupied levels
        mape_z = mape.median(dim='cloud', skipna=True)
        overall = float(mape_z.median(skipna=True).values)
        print("Per-z median MAPE (%):", np.array2string(mape_z.values, precision=1, separator=", "))
        print("Overall median MAPE (occupied levels):", overall)
        coverage = present.mean(dim='cloud').rename('coverage')
        print("Coverage per z:", np.array2string(coverage.values, precision=2, separator=", "))
        # Lifetime diagnostics
        try:
            train_tc = np.median([float(ds.attrs['T_c']) for ds in train])
            test_tc = [float(ds.attrs['T_c']) for ds in test]
            print("Train median T_c [s]:", train_tc)
            print("Test T_c [s]:", test_tc)
        except Exception:
            pass
        try:
            cz_vals = c_z.values if isinstance(c_z, xr.DataArray) else c_z['c'].values
            print("Median c(z) [s]:", float(np.nanmedian(cz_vals)))
        except Exception:
            pass
        print(f"[run] eval done in {time.time()-t_eval:.1f}s", flush=True)

    # Save c(z)
    default_out = 'analysis/lifetime_massflux/cz_estimate.nc'
    if args.out == default_out:
        # If user runs from inside analysis/lifetime_massflux, avoid duplicating the path.
        out_path = (Path(__file__).resolve().parent / 'cz_estimate.nc')
        print(f"[run] resolving default --out to {out_path}", flush=True)
    else:
        out_path = Path(args.out).expanduser()
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] saving c(z) to {out_path} ...", flush=True)
    t_save = time.time()
    c_z.to_dataset(name='c').to_netcdf(str(out_path))
    print(f"[run] saved in {time.time()-t_save:.1f}s", flush=True)
    print(f"[run] total time {time.time()-t_all:.1f}s. bye.", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Lifetime mass-flux analysis on cloud_results.nc')
    p.add_argument('--nc', default='../../cloud_results_0.000001_30min.nc', help='Path to cloud_results.nc')
    p.add_argument('--dt', type=float, default=60.0, help='Timestep duration in seconds')
    p.add_argument('--rho0', default='raw', choices=['raw','env','var'], help='Reference density source')
    p.add_argument('--raw_base', default='/Users/jure/PhD/coding/RICO_1hr/', help='Base path to raw LES files for --rho0 raw')
    p.add_argument('--rho_sample_frac', type=float, default=0.001, help='Fraction of horizontal domain to sample for rho0 (random)')
    p.add_argument('--rho_time_max', type=int, default=1, help='Max number of timesteps to use for rho0')
    p.add_argument('--min_timesteps', type=int, default=3, help='Min active timesteps per track')
    p.add_argument('--test_last', type=int, default=5, help='Number of clouds for test set (last)')
    p.add_argument('--no_smooth', action='store_true', help='Disable z-smoothing for c(z)')
    p.add_argument('--use_field_density', action='store_true', help='Fit c(z) using J_rho (instantaneous density) instead of J')
    p.add_argument('--include_partial', action='store_true', help='Include partial (invalid) tracks')
    p.add_argument('--out', default='analysis/lifetime_massflux/cz_estimate.nc', help='Output NetCDF for c(z)')
    main(p.parse_args())
