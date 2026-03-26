#!/usr/bin/env python3
"""
3D visualisation of a tracked cloud at every minute of its lifetime,
plus a configurable margin before birth and after death.

Produces a multi-panel figure (one subplot per timestep) showing the
isosurface of liquid water content in a box that follows the cloud
centroid.  The cloud surface is extracted with marching cubes and
coloured by height.

Usage
-----
    # Default cloud (#28244) with 2-min margin — interactive window
    python cloud_lifecycle_3d.py

    # Save as tiled PNG
    python cloud_lifecycle_3d.py --cloud 28244 --save

    # Save as animated GIF (one frame per timestep)
    python cloud_lifecycle_3d.py --cloud 28244 --gif

    # Custom margin and search radius
    python cloud_lifecycle_3d.py --cloud 28244 --gif --margin 3 --radius 3000
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from netCDF4 import Dataset
from skimage import measure

# ── Paths (edit for your machine) ─────────────────────────────────────────
TRACKING_FILE = "../output_archive/cloud_results_120mins.nc"
LES_FILE = "/data/jz557/rico_case/july7/rico.l.nc"
L_VAR = "l"
L_THRESHOLD = 1e-5          # same as tracking config
OUTPUT_DIR = "cloud_3d_lifecycle"
SEARCH_RADIUS = 2000        # metres around centroid


def load_tracking_data(cloud_index):
    """Return centroid coordinates and active timesteps for one cloud."""
    with Dataset(TRACKING_FILE, "r") as ds:
        loc_x = np.ma.filled(ds.variables["location_x"][cloud_index], np.nan)
        loc_y = np.ma.filled(ds.variables["location_y"][cloud_index], np.nan)
        loc_z = np.ma.filled(ds.variables["location_z"][cloud_index], np.nan)
        valid_track = int(ds.variables["valid_track"][cloud_index])
    active = np.isfinite(loc_x)
    timesteps = np.where(active)[0]
    centroids = {int(t): (float(loc_x[t]), float(loc_y[t]), float(loc_z[t]))
                 for t in timesteps}
    return timesteps, centroids, valid_track


def load_grid():
    """Return the coordinate arrays from the LES file."""
    with Dataset(LES_FILE, "r") as ds:
        xt = np.asarray(ds.variables["xt"][:], dtype=float)
        yt = np.asarray(ds.variables["yt"][:], dtype=float)
        zt = np.asarray(ds.variables["zt"][:], dtype=float)
        n_times = ds.dimensions["time"].size
    return xt, yt, zt, n_times


def extract_isosurface(t, xt, yt, zt, centroid, search_radius):
    """
    Extract cloud isosurface vertices + faces in a box around *centroid*.
    Returns (verts, faces) in physical coordinates, or (None, None).
    """
    cx, cy, _ = centroid
    dx = float(xt[1] - xt[0])
    dy = float(yt[1] - yt[0])
    dz = float(zt[1] - zt[0])

    ix = int(np.abs(xt - cx).argmin())
    iy = int(np.abs(yt - cy).argmin())
    rx = int(np.ceil(search_radius / dx))
    ry = int(np.ceil(search_radius / dy))

    x0 = max(0, ix - rx);  x1 = min(len(xt), ix + rx)
    y0 = max(0, iy - ry);  y1 = min(len(yt), iy + ry)

    with Dataset(LES_FILE, "r") as ds:
        sub = np.asarray(ds.variables[L_VAR][t, :, y0:y1, x0:x1])

    mask = (sub >= L_THRESHOLD).astype(float)
    if mask.max() == 0:
        return None, None

    try:
        verts, faces, _, _ = measure.marching_cubes(
            mask, level=0.5, spacing=(dz, dy, dx)
        )
    except Exception:
        return None, None

    # Shift to world coordinates
    verts[:, 0] += zt[0]
    verts[:, 1] += yt[y0]
    verts[:, 2] += xt[x0]
    return verts, faces


def draw_timestep(ax, t, verts, faces, centroid, z_lim, view_range, label):
    """Render one isosurface panel."""
    cx, cy, cz = centroid
    ax.set_xlim(cx - view_range / 2, cx + view_range / 2)
    ax.set_ylim(cy - view_range / 2, cy + view_range / 2)
    ax.set_zlim(*z_lim)

    if verts is not None and faces is not None:
        # Colour by height
        norm = Normalize(vmin=z_lim[0], vmax=z_lim[1])
        face_z = verts[faces].mean(axis=1)[:, 0]
        colours = plt.cm.viridis(norm(face_z))
        mesh = Poly3DCollection(
            verts[faces][:, :, [2, 1, 0]],   # swap to (x, y, z) for mpl
            facecolor=colours, edgecolor="none", alpha=0.7,
        )
        ax.add_collection3d(mesh)

    # Mark centroid
    ax.scatter(cx, cy, cz, color="red", s=40, zorder=5)

    ax.set_xlabel("X (m)", fontsize=7)
    ax.set_ylabel("Y (m)", fontsize=7)
    ax.set_zlabel("Z (m)", fontsize=7)
    ax.set_title(label, fontsize=9)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=25, azim=-60)


def main():
    parser = argparse.ArgumentParser(
        description="3-D lifecycle visualisation of a tracked cloud."
    )
    parser.add_argument(
        "--cloud", type=int, default=28244,
        help="Cloud index in the tracking NetCDF (default: 28244)",
    )
    parser.add_argument(
        "--margin", type=int, default=2,
        help="Extra timesteps before birth / after death (default: 2)",
    )
    parser.add_argument(
        "--radius", type=float, default=SEARCH_RADIUS,
        help=f"Search radius around centroid in metres (default: {SEARCH_RADIUS})",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save tiled figure to PNG instead of displaying interactively.",
    )
    parser.add_argument(
        "--gif", action="store_true",
        help="Save an animated GIF (one frame per timestep).",
    )
    parser.add_argument(
        "--fps", type=int, default=2,
        help="Frames per second for the GIF (default: 2).",
    )
    args = parser.parse_args()

    cloud_idx = args.cloud
    margin = args.margin
    search_radius = args.radius

    print(f"Cloud index : {cloud_idx}")
    print(f"Margin      : ±{margin} timesteps")

    # ── Load tracking info ────────────────────────────────────────────────
    timesteps, centroids, valid = load_tracking_data(cloud_idx)
    if len(timesteps) == 0:
        sys.exit(f"Cloud {cloud_idx} has no active timesteps.")
    print(f"Valid track : {'yes' if valid == 1 else 'no (tainted)'}")
    print(f"Active steps: {timesteps[0]}–{timesteps[-1]} "
          f"({len(timesteps)} steps = {len(timesteps)} min)")

    # ── Grid ──────────────────────────────────────────────────────────────
    xt, yt, zt, n_les_times = load_grid()

    # Build frame list with margin
    first = max(0, int(timesteps[0]) - margin)
    last = min(n_les_times - 1, int(timesteps[-1]) + margin)
    frames = list(range(first, last + 1))
    n_frames = len(frames)
    print(f"Frames      : {first}–{last} ({n_frames} panels)")

    # Fallback centroids for margin frames
    first_centroid = centroids[int(timesteps[0])]
    last_centroid = centroids[int(timesteps[-1])]

    # ── Extract all isosurfaces ───────────────────────────────────────────
    surfaces = {}
    all_z = []
    for i, t in enumerate(frames):
        centroid = centroids.get(t, first_centroid if t < timesteps[0] else last_centroid)
        pct = (i + 1) / n_frames * 100
        bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
        print(f"\r  [{bar}] {pct:5.1f}%  frame {t:>3d}/{frames[-1]} ", end="", flush=True)
        verts, faces = extract_isosurface(t, xt, yt, zt, centroid, search_radius)
        surfaces[t] = (verts, faces, centroid)
        if verts is not None:
            all_z.extend([verts[:, 0].min(), verts[:, 0].max()])
    print()  # newline after progress bar

    if all_z:
        z_lim = (min(all_z) - 200, max(all_z) + 200)
    else:
        z_lim = (0, 3000)

    # ── Output ─────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.gif:
        # ── Animated GIF: one frame per timestep ──────────────────────────
        from PIL import Image
        import io

        pil_frames = []
        print("Rendering GIF frames...")
        for i, t in enumerate(frames):
            pct = (i + 1) / n_frames * 100
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(f"\r  [{bar}] {pct:5.1f}%  frame {i+1}/{n_frames} ",
                  end="", flush=True)

            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            verts, faces, centroid = surfaces[t]

            if t < timesteps[0]:
                phase = "pre-birth"
            elif t > timesteps[-1]:
                phase = "post-death"
            else:
                age = t - int(timesteps[0])
                phase = f"age {age} min"

            label = f"Cloud #{cloud_idx}   t = {t}  ({phase})"
            draw_timestep(ax, t, verts, faces, centroid, z_lim,
                          search_radius, label)

            # Render to in-memory PNG → PIL Image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            pil_frames.append(Image.open(buf).copy())
            buf.close()

        print()
        gif_path = os.path.join(
            OUTPUT_DIR, f"cloud_{cloud_idx}_lifecycle_3d.gif")
        duration_ms = int(1000 / args.fps)
        pil_frames[0].save(
            gif_path, save_all=True, append_images=pil_frames[1:],
            duration=duration_ms, loop=0,
        )
        print(f"Saved → {gif_path}  ({len(pil_frames)} frames, {args.fps} fps)")

    else:
        # ── Tiled PNG (original behaviour) ────────────────────────────────
        ncols = min(6, n_frames)
        nrows = int(np.ceil(n_frames / ncols))
        fig = plt.figure(figsize=(ncols * 4, nrows * 4))
        fig.suptitle(f"Cloud #{cloud_idx}  —  3-D lifecycle  "
                     f"(margin ±{margin} min)", fontsize=14, y=0.98)

        print("Rendering panels...")
        for i, t in enumerate(frames):
            pct = (i + 1) / n_frames * 100
            bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
            print(f"\r  [{bar}] {pct:5.1f}%  panel {i+1}/{n_frames} ",
                  end="", flush=True)

            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            verts, faces, centroid = surfaces[t]

            if t < timesteps[0]:
                phase = "pre-birth"
            elif t > timesteps[-1]:
                phase = "post-death"
            else:
                age = t - int(timesteps[0])
                phase = f"age {age} min"

            label = f"t = {t}  ({phase})"
            draw_timestep(ax, t, verts, faces, centroid, z_lim,
                          search_radius, label)
        print()

        fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02,
                            wspace=0.15, hspace=0.25)

        if args.save:
            out_path = os.path.join(
                OUTPUT_DIR, f"cloud_{cloud_idx}_lifecycle_3d.png")
            print(f"Saving {nrows}×{ncols} 3-D figure...", flush=True)
            fig.savefig(out_path, dpi=100)
            print(f"Saved → {out_path}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
