import argparse
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


def compute_track_lifetime_steps(age_row: np.ndarray) -> float:
    """
    Compute lifetime directly from the NetCDF 'age' (time,) where inactive entries are -1.
    Age starts at 0 at birth and increments by 1 each timestep, so lifetime_steps = max(age)+1.
    Returns 0.0 for tracks with no valid ages.
    """
    valid = age_row >= 0
    if not np.any(valid):
        return 0.0
    max_age = np.nanmax(age_row[valid])
    lifetime_steps = float(max_age) + 1.0
    return lifetime_steps


def compute_track_nip_acc_stat(nip_acc_block: np.ndarray, time_agg: str = 'median') -> float:
    """
    Compute a single scalar per track from nip_acc_per_level[time, level].
    Approach: vertically integrate per time (nan-sum), then aggregate across time
    using 'median', 'mean', 'p75', 'p90', or 'max'.
    Returns np.nan if nothing finite.
    """
    if nip_acc_block.size == 0:
        return np.nan
    # Vertical sum per time
    with np.errstate(invalid='ignore'):
        v_int = np.nansum(nip_acc_block, axis=1)
    finite = np.isfinite(v_int)
    if not np.any(finite):
        return np.nan
    v = v_int[finite]
    if time_agg == 'mean':
        return float(np.nanmean(v))
    if time_agg == 'p75':
        return float(np.nanpercentile(v, 75))
    if time_agg == 'p90':
        return float(np.nanpercentile(v, 90))
    if time_agg == 'max':
        return float(np.nanmax(v))
    # default median
    return float(np.nanmedian(v))


def load_nip_vs_lifetime(nc_path: str, only_valid: bool = True,
                          min_steps: int = 1,
                          early_steps: int = 0,
                          time_agg: str = 'median') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load per-track lifetime and median vertically-integrated accumulated NIP from NetCDF.

    - lifetime is computed from 'age' as (max(age)+1) steps (no minute conversion)
    - accumulated NIP is computed as aggregate over time (per time sum over levels) of nip_acc_per_level
    - filters to valid tracks if only_valid is True (uses 'valid_track' == 1)
    - filters to tracks with at least min_steps presence
    - if early_steps > 0, use only the first early_steps of the track (based on age)
    Returns (nip_acc_median, lifetime_minutes) arrays of equal length.
    """
    with Dataset(nc_path, 'r') as ds:
        valid_track = ds.variables.get('valid_track')
        age = ds.variables['age']  # (track, time)
        nip_acc = ds.variables['nip_acc_per_level']  # (track, time, level)

        n_tracks = age.shape[0]
        nip_vals = []
        life_vals = []

        for i in range(n_tracks):
            if only_valid and valid_track is not None:
                vt = int(valid_track[i])
                if vt != 1:
                    continue

            age_row = np.array(age[i, :])
            present = age_row >= 0
            if np.count_nonzero(present) < min_steps:
                continue

            # Lifetime in steps (from 'age' only)
            life_steps = compute_track_lifetime_steps(age_row)
            if life_steps <= 0:
                continue

            # Slice the time window: first early_steps if requested, else full active window
            active_idx = np.where(present)[0]
            if early_steps and early_steps > 0:
                # Select indices where age <= early_steps-1 to ensure exact early window
                window_mask = present & (age_row <= (early_steps - 1))
                win_idx = np.where(window_mask)[0]
                if win_idx.size == 0:
                    continue
                t_start = win_idx.min()
                t_end = win_idx.max() + 1
            else:
                t_start = active_idx.min()
                t_end = active_idx.max() + 1

            nip_block = np.array(nip_acc[i, t_start:t_end, :])  # (time_window, level)

            # Some datasets may have all-NaN NIP; skip those
            nip_stat = compute_track_nip_acc_stat(nip_block, time_agg=time_agg)
            if not np.isfinite(nip_stat):
                continue

            nip_vals.append(nip_stat)
            life_vals.append(life_steps)

    return np.asarray(nip_vals), np.asarray(life_vals)


def binned_xy(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_bin_centers, y_mean) using quantile-based bins for robust trend visualization."""
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return np.array([]), np.array([])
    # Quantile bin edges
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs)
    # Ensure strictly increasing edges (handle duplicates)
    edges = np.unique(edges)
    if edges.size < 3:
        return np.array([]), np.array([])
    centers = []
    means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x <= hi)
        if np.count_nonzero(sel) == 0:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(np.nanmean(y[sel]))
    return np.asarray(centers), np.asarray(means)


def plot_nip_vs_lifetime(nc_path: str, out_path: str,
                          only_valid: bool = True, min_steps: int = 2,
                          alpha: float = 0.2, bins: int = 20,
                          unit: str = 'steps', min_lifetime: float = 0.0,
                          early_steps: int = 0, time_agg: str = 'median') -> None:
    # Load NIP statistic (first) and lifetime in steps (second)
    nip_vals, life_steps = load_nip_vs_lifetime(
        nc_path,
        only_valid=only_valid,
        min_steps=min_steps,
        early_steps=early_steps,
        time_agg=time_agg,
    )
    y = nip_vals
    x_steps = life_steps

    # Convert x if requested
    if unit == 'minutes':
        # Need timestep duration from file if present; otherwise assume 60s
        dt_seconds = 60.0
        try:
            with Dataset(nc_path, 'r') as ds:
                # No explicit dt stored; keep default 60 s unless you store it later
                pass
        except Exception:
            pass
        x = x_steps * (dt_seconds / 60.0)
        x_label = 'Cloud lifetime (minutes)'
    else:
        x = x_steps
        x_label = 'Cloud lifetime (timesteps)'

    # Apply minimum lifetime filter in the plotted unit
    if min_lifetime is not None and min_lifetime > 0:
        sel = np.isfinite(x) & np.isfinite(y) & (x >= float(min_lifetime))
        x = x[sel]
        y = y[sel]

    # Remove non-positive y values for log-scale plotting
    pos_mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[pos_mask]
    y = y[pos_mask]

    if x.size == 0:
        raise RuntimeError("No valid tracks found for plotting. Check filters or input file.")

    # Spearman correlation (monotonic relationship robustness)
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(x, y)
        corr_text = f"Spearman r={rho:.2f}, p={pval:.2e}"
    except Exception:
        corr_text = None

    # Binned trend
    bx, by = binned_xy(x, y, n_bins=bins)

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=8, alpha=alpha, edgecolor='none', color='tab:blue')
    plt.yscale('log')
    if bx.size and by.size:
        order = np.argsort(bx)
        plt.plot(bx[order], by[order], color='tab:red', linewidth=2, label='binned mean')
        plt.legend(frameon=False)
    # Lifetime on x-axis
    plt.xlabel(x_label)
    # Dynamic y-label reflecting aggregation and early window
    y_agg_label = {
        'median': 'Median',
        'mean': 'Mean',
        'p75': '75th pct',
        'p90': '90th pct',
        'max': 'Max',
    }.get(time_agg, 'Median')
    if early_steps and early_steps > 0:
        plt.ylabel(f"{y_agg_label} of vertically-integrated accumulated NIP (first {early_steps} steps)")
        title = f"Accumulated NIP (early {early_steps}) vs Lifetime (valid tracks)"
    else:
        plt.ylabel(f"{y_agg_label} vertically-integrated accumulated NIP per track")
        title = 'Accumulated NIP vs Lifetime (valid tracks)'
    if corr_text:
        title += f"\n{corr_text}"
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def debug_summary(nc_path: str, only_valid: bool, min_steps: int, unit: str, min_lifetime: float,
                  early_steps: int, time_agg: str) -> None:
    """Print a brief, stepwise filter summary for troubleshooting."""
    try:
        with Dataset(nc_path, 'r') as ds:
            valid_track = ds.variables.get('valid_track')
            age = ds.variables['age']
            nip_acc = ds.variables['nip_acc_per_level']

            n_tracks = age.shape[0]
            idx_all = np.arange(n_tracks)

            # Valid filter
            if only_valid and valid_track is not None:
                idx_valid = idx_all[np.array(valid_track[:]) == 1]
            else:
                idx_valid = idx_all

            # Min steps filter
            present_counts = np.sum(np.array(age[:]) >= 0, axis=1)
            idx_steps = idx_valid[present_counts[idx_valid] >= min_steps]

            # Lifetimes in steps
            life_steps = np.zeros(n_tracks, dtype=float)
            for i in idx_steps:
                row = np.array(age[i, :])
                mask = row >= 0
                if np.any(mask):
                    life_steps[i] = float(np.nanmax(row[mask])) + 1.0
            if unit == 'minutes':
                dt_seconds = 60.0
                life_unit = life_steps * (dt_seconds / 60.0)
            else:
                life_unit = life_steps

            # Min lifetime filter
            idx_life = idx_steps[life_unit[idx_steps] >= (min_lifetime or 0.0)]

            # Finite NIP filter using the chosen early_steps window and aggregator
            fin_idx = []
            for i in idx_life:
                row = np.array(age[i, :])
                mask = row >= 0
                if not np.any(mask):
                    continue
                if early_steps and early_steps > 0:
                    win_mask = mask & (row <= (early_steps - 1))
                    if not np.any(win_mask):
                        continue
                    t0, t1 = np.where(win_mask)[0][[0, -1]]
                else:
                    t0, t1 = np.where(mask)[0][[0, -1]]
                block = np.array(nip_acc[i, t0:t1 + 1, :])
                with np.errstate(invalid='ignore'):
                    v_int = np.nansum(block, axis=1)
                # apply agg
                finite = np.isfinite(v_int)
                if not np.any(finite):
                    continue
                v = v_int[finite]
                if time_agg == 'mean':
                    val = np.nanmean(v)
                elif time_agg == 'p75':
                    val = np.nanpercentile(v, 75)
                elif time_agg == 'p90':
                    val = np.nanpercentile(v, 90)
                elif time_agg == 'max':
                    val = np.nanmax(v)
                else:
                    val = np.nanmedian(v)
                if np.isfinite(val):
                    fin_idx.append(i)

            print("Filter summary:")
            print(f"  total tracks: {n_tracks}")
            print(f"  after valid filter: {idx_valid.size}")
            print(f"  after min_steps>={min_steps}: {idx_steps.size}")
            print(f"  after min_lifetime>={min_lifetime} {unit}: {len(idx_life)}")
            print(f"  after finite NIP: {len(fin_idx)}")
            if len(idx_valid) > 0:
                max_life_valid = np.nanmax(life_steps[idx_valid]) if np.any(life_steps[idx_valid] > 0) else 0
                print(f"  max lifetime among valid-filtered (steps): {max_life_valid:.0f}")
    except Exception as e:
        print(f"Debug summary failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Plot relationship between accumulated NIP and cloud lifetime.')
    parser.add_argument('--nc', default='../cloud_results.nc', help='Path to NetCDF results (default: cloud_results.nc)')
    parser.add_argument('--out', default='analysis_output/nip_vs_lifetime.png', help='Output PNG path')
    parser.add_argument('--all', action='store_true', help='Include partial/tainted tracks as well')
    parser.add_argument('--min_steps', type=int, default=2, help='Minimum timesteps present per track (default: 2)')
    parser.add_argument('--unit', choices=['steps', 'minutes'], default='steps', help='X-axis lifetime units (default: steps)')
    parser.add_argument('--min_lifetime', type=float, default=None, help='Minimum lifetime threshold in chosen unit (default: prompt).')
    parser.add_argument('--no-prompt', action='store_true', help='Disable interactive prompt for min_lifetime; default to 0 if not provided.')
    parser.add_argument('--debug', action='store_true', help='Print filter summary for troubleshooting.')
    parser.add_argument('--early_steps', type=int, default=0, help='Use only first N steps for NIP aggregation (0=all).')
    parser.add_argument('--agg', choices=['median','mean','p75','p90','max'], default='mean', help='Time aggregation for NIP over the window.')
    args = parser.parse_args()

    only_valid = not args.all
    # Determine minimum lifetime threshold
    min_lifetime = args.min_lifetime
    if min_lifetime is None and not args.no_prompt:
        try:
            user = input(f"Minimum cloud lifetime in {args.unit} (empty for 0): ").strip()
            min_lifetime = float(user) if user else 0.0
        except (EOFError, ValueError):
            min_lifetime = 0.0
    if min_lifetime is None:
        min_lifetime = 0.0

    if args.debug:
        debug_summary(args.nc, only_valid, args.min_steps, args.unit, min_lifetime, args.early_steps, args.agg)

    plot_nip_vs_lifetime(
        args.nc,
        args.out,
        only_valid=only_valid,
        min_steps=args.min_steps,
        unit=args.unit,
        min_lifetime=min_lifetime,
        early_steps=args.early_steps,
        time_agg=args.agg,
    )
    print(f"Saved plot to {args.out}")


if __name__ == '__main__':
    main()
