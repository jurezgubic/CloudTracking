"""
Very simple class labels from lifetime-mean profiles (optional).

# physics: classify cloud by geometric depth based on where it exists.
# NOTE: placeholder logic now (always returns 0), but shows how to use S_a(z).
"""

from __future__ import annotations
import xarray as xr
import numpy as np

def depth_quartile(ds_reduced: xr.Dataset) -> int:
    z = ds_reduced["z"].values
    # present where time-integrated area S_a > 0
    present = xr.where(ds_reduced["S_a"] > 0, 1, 0)
    if present.sum() == 0:
        return -1
    idx = np.where(present.values > 0)[0]
    depth = z[idx[-1]] - z[idx[0]]
    # Bin by quartiles of depth requires global stats; placeholder returns 0
    return 0
