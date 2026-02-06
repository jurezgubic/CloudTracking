"""
LBA Cloud 3D Visualizer

Creates 3D animated visualizations of tracked clouds from LBA MONC simulation.
Follows a single cloud through its lifecycle, showing the cloud surface evolution.

Usage:
    python lba_cloud_3d_visualizer.py
"""

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from skimage import measure
import os
import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.adapters.monc_adapter import MONCAdapter

# =============================================================================
# CONFIGURATION - Adjust these parameters
# =============================================================================

# Cloud tracking output file
TRACKING_OUTPUT = '../../cloud_results_lba.nc'

# MONC data configuration (same as main.py)
MONC_CONFIG = {
    'data_format': 'MONC',
    'monc_data_path': '/Users/jure/PhD/coding/LBA_sample_data/jun10',
    'monc_config_file': '/Users/jure/PhD/coding/LBA_sample_data/jun10/lba_config.mcf',
    'monc_file_pattern': '3dfields_ts_{time}.nc',
}

# Visualization parameters
MIN_LIFETIME_TIMESTEPS = 3      # Minimum cloud lifetime to visualize
L_CONDITION = 1e-5              # Liquid water threshold for isosurface (kg/kg)
SEARCH_RADIUS = 5000            # Search radius around cloud centroid (m)
VIEW_RANGE = 2500               # Horizontal view range (m) - smaller = more zoomed in
OUTPUT_FOLDER = 'cloud_3d_visualizations'

# Simulation parameters
DT = 180  # seconds between timesteps

# =============================================================================
# END CONFIGURATION
# =============================================================================

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def find_long_lived_clouds(min_lifetime=MIN_LIFETIME_TIMESTEPS):
    """Find clouds with sufficient lifetime for visualization."""
    print(f"Finding clouds with lifetime >= {min_lifetime} timesteps...")
    
    with Dataset(TRACKING_OUTPUT, 'r') as nc:
        valid_track = nc.variables['valid_track'][:]
        size = np.ma.filled(nc.variables['size'][:], np.nan)
        max_height = np.ma.filled(nc.variables['max_height'][:], np.nan)
        cloud_base_height = np.ma.filled(nc.variables['cloud_base_height'][:], np.nan)
        loc_x = np.ma.filled(nc.variables['location_x'][:], np.nan)
        loc_y = np.ma.filled(nc.variables['location_y'][:], np.nan)
        loc_z = np.ma.filled(nc.variables['location_z'][:], np.nan)
        max_w = np.ma.filled(nc.variables['max_w'][:], np.nan)
        merges_count = np.ma.filled(nc.variables['merges_count'][:], 0)
        splits_count = np.ma.filled(nc.variables['splits_count'][:], 0)
        
        n_tracks = size.shape[0]
        
    candidate_clouds = []
    
    for idx in range(n_tracks):
        # Find timesteps where cloud exists
        valid_timesteps = np.where(np.isfinite(size[idx]) & (size[idx] > 0))[0]
        lifetime = len(valid_timesteps)
        
        if lifetime < min_lifetime:
            continue
        
        first_timestep = valid_timesteps[0]
        last_timestep = valid_timesteps[-1]
        
        # Get cloud statistics
        max_size = np.nanmax(size[idx])
        max_top = np.nanmax(max_height[idx])
        max_w_val = np.nanmax(max_w[idx])
        total_merges = int(np.nansum(merges_count[idx]))
        total_splits = int(np.nansum(splits_count[idx]))
        is_valid = valid_track[idx] == 1
        
        # Store centroids for each valid timestep
        centroids = [(float(loc_x[idx, t]), float(loc_y[idx, t]), float(loc_z[idx, t])) 
                     for t in valid_timesteps]
        
        candidate_clouds.append({
            'track_id': idx,
            'lifetime': lifetime,
            'first_timestep': int(first_timestep),
            'last_timestep': int(last_timestep),
            'valid_timesteps': valid_timesteps.tolist(),
            'max_size': float(max_size),
            'max_height': float(max_top),
            'max_w': float(max_w_val),
            'merges': total_merges,
            'splits': total_splits,
            'is_valid': is_valid,
            'centroids': centroids,
            'base_heights': cloud_base_height[idx, valid_timesteps].tolist(),
        })
    
    # Sort by lifetime (longest first)
    candidate_clouds.sort(key=lambda x: x['lifetime'], reverse=True)
    
    print(f"Found {len(candidate_clouds)} clouds with lifetime >= {min_lifetime} timesteps")
    
    # Display cloud information
    print("\nCandidate clouds for visualization:")
    print("-" * 100)
    print(f"{'Idx':<4} {'Track':<8} {'Life':<6} {'Start':<6} {'End':<6} {'MaxSize':<10} {'MaxH(km)':<10} {'MaxW':<8} {'Merge':<6} {'Split':<6} {'Valid':<6}")
    print("-" * 100)
    
    for i, cloud in enumerate(candidate_clouds[:20]):  # Show top 20
        print(f"{i:<4} {cloud['track_id']:<8} {cloud['lifetime']:<6} "
              f"{cloud['first_timestep']:<6} {cloud['last_timestep']:<6} "
              f"{cloud['max_size']:<10.0f} {cloud['max_height']/1000:<10.1f} "
              f"{cloud['max_w']:<8.1f} {cloud['merges']:<6} {cloud['splits']:<6} "
              f"{'Yes' if cloud['is_valid'] else 'No':<6}")
    
    if len(candidate_clouds) > 20:
        print(f"... and {len(candidate_clouds) - 20} more clouds")
    
    return candidate_clouds


def extract_cloud_surface(l_data, xt, yt, zt, centroid, search_radius=SEARCH_RADIUS):
    """
    Extract the cloud surface points around a centroid using marching cubes.
    
    Args:
        l_data: 3D liquid water content array (z, y, x)
        xt, yt, zt: 1D coordinate arrays
        centroid: (x, y, z) position of cloud centroid
        search_radius: Maximum radius around centroid to search (m)
        
    Returns:
        verts, faces: Vertices and faces of the cloud surface mesh, or (None, None)
    """
    cx, cy, cz = centroid
    
    # Handle non-uniform grid spacing
    dx = xt[1] - xt[0] if len(xt) > 1 else 200.0
    dy = yt[1] - yt[0] if len(yt) > 1 else 200.0
    dz_mean = np.mean(np.diff(zt)) if len(zt) > 1 else 50.0
    
    # Find grid indices corresponding to centroid
    x_idx = np.abs(xt - cx).argmin()
    y_idx = np.abs(yt - cy).argmin()
    z_idx = np.abs(zt - cz).argmin()
    
    # Define region of interest around centroid
    x_radius = int(np.ceil(search_radius / dx))
    y_radius = int(np.ceil(search_radius / dy))
    z_radius = int(np.ceil(search_radius / dz_mean))
    
    # Calculate bounds with domain boundary checks
    x_min = max(0, x_idx - x_radius)
    x_max = min(len(xt) - 1, x_idx + x_radius)
    y_min = max(0, y_idx - y_radius)
    y_max = min(len(yt) - 1, y_idx + y_radius)
    z_min = max(0, z_idx - z_radius)
    z_max = min(len(zt) - 1, z_idx + z_radius)
    
    # Extract subvolume (l_data is z, y, x order)
    subvolume = l_data[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    
    # Create binary mask for cloud
    cloud_mask = subvolume >= L_CONDITION
    
    # Skip if no cloud is found in this region
    if not np.any(cloud_mask):
        return None, None
    
    # Get local grid spacings for marching cubes
    local_dz = zt[min(z_min+1, len(zt)-1)] - zt[z_min] if z_max > z_min else dz_mean
    local_dy = yt[min(y_min+1, len(yt)-1)] - yt[y_min] if y_max > y_min else dy
    local_dx = xt[min(x_min+1, len(xt)-1)] - xt[x_min] if x_max > x_min else dx
    
    try:
        # Run marching cubes
        verts, faces, _, _ = measure.marching_cubes(
            cloud_mask.astype(float), 
            level=0.5,
            spacing=(local_dz, local_dy, local_dx)
        )
        
        # Transform vertices to world coordinates
        verts[:, 0] += zt[z_min]  # z
        verts[:, 1] += yt[y_min]  # y
        verts[:, 2] += xt[x_min]  # x
        
        return verts, faces
    
    except Exception as e:
        print(f"  Warning: Could not extract isosurface: {e}")
        return None, None


def visualize_cloud_evolution(cloud_info, adapter):
    """Create 3D animated visualization of cloud evolution."""
    track_id = cloud_info['track_id']
    print(f"\nVisualizing Cloud {track_id}...")
    
    # Get grid info from adapter
    grid_info = adapter.get_grid_info()
    xt = grid_info['xt']
    yt = grid_info['yt']
    zt = grid_info['zt']
    
    # Get timestep info
    all_times = adapter.get_timestep_times()
    n_available = adapter.get_total_timesteps()
    
    # Cloud lifecycle bounds
    first_ts = cloud_info['first_timestep']
    last_ts = cloud_info['last_timestep']
    valid_timesteps = cloud_info['valid_timesteps']
    
    # Extend visualization: 1 timestep before and after (if available)
    vis_start = max(0, first_ts - 1)
    vis_end = min(n_available - 1, last_ts + 1)
    
    print(f"  Cloud lifetime: timesteps {first_ts} to {last_ts}")
    print(f"  Visualization range: timesteps {vis_start} to {vis_end}")
    
    # Create centroid lookup
    centroid_map = {t: c for t, c in zip(valid_timesteps, cloud_info['centroids'])}
    
    # Extract cloud surfaces for each timestep
    print("  Extracting cloud surfaces...")
    cloud_surfaces = []
    
    for t_idx in range(vis_start, vis_end + 1):
        print(f"    Timestep {t_idx} (t={all_times[t_idx]:.0f}s)...", end=' ')
        
        # Load 3D data for this timestep
        data = adapter.load_timestep(t_idx)
        l_data = data['l']  # liquid water content (z, y, x)
        
        # Get centroid (use nearest valid if outside cloud's lifetime)
        if t_idx in centroid_map:
            centroid = centroid_map[t_idx]
        elif t_idx < first_ts:
            centroid = centroid_map[valid_timesteps[0]]
        else:
            centroid = centroid_map[valid_timesteps[-1]]
        
        # Extract surface
        verts, faces = extract_cloud_surface(l_data, xt, yt, zt, centroid)
        
        if verts is not None:
            print(f"{len(verts)} vertices")
        else:
            print("no surface")
        
        cloud_surfaces.append({
            'timestep': t_idx,
            'time_s': all_times[t_idx],
            'verts': verts,
            'faces': faces,
            'centroid': centroid,
            'in_lifecycle': t_idx in valid_timesteps,
        })
    
    # Check if we have any surfaces
    if not any(s['verts'] is not None for s in cloud_surfaces):
        print("  ERROR: No cloud surfaces found!")
        return
    
    # Calculate z range for consistent vertical scale
    valid_surfaces = [s for s in cloud_surfaces if s['verts'] is not None]
    all_verts = np.vstack([s['verts'] for s in valid_surfaces])
    z_min = max(0, np.min(all_verts[:, 0]) - 500)
    z_max = np.max(all_verts[:, 0]) + 500
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def update_frame(frame_idx):
        ax.clear()
        
        surf = cloud_surfaces[frame_idx]
        t_idx = surf['timestep']
        time_s = surf['time_s']
        time_min = time_s / 60
        verts = surf['verts']
        faces = surf['faces']
        centroid = surf['centroid']
        in_lifecycle = surf['in_lifecycle']
        
        # View center follows cloud
        cx, cy, cz = centroid
        
        if verts is not None and faces is not None:
            # Color based on lifecycle phase
            if not in_lifecycle:
                color = 'gray'
                alpha = 0.4
            else:
                # Age-based coloring
                age = t_idx - first_ts
                cmap = plt.cm.viridis
                color = cmap(age / max(1, last_ts - first_ts))
                alpha = 0.8
            
            # Plot surface
            ax.plot_trisurf(
                verts[:, 2], verts[:, 1], verts[:, 0],  # x, y, z
                triangles=faces,
                color=color,
                alpha=alpha,
                edgecolor='none'
            )
        
        # Mark centroid
        ax.scatter([cx], [cy], [cz], color='red', s=100, marker='o', 
                   edgecolor='black', zorder=10)
        
        # Add cloud base plane (if we have base height)
        if in_lifecycle:
            base_idx = valid_timesteps.index(t_idx)
            base_height = cloud_info['base_heights'][base_idx]
            
            # Small plane at cloud base
            plane_size = VIEW_RANGE / 4
            xx, yy = np.meshgrid(
                [cx - plane_size/2, cx + plane_size/2],
                [cy - plane_size/2, cy + plane_size/2]
            )
            zz = np.ones_like(xx) * base_height
            ax.plot_surface(xx, yy, zz, color='green', alpha=0.2)
        
        # Labels and title
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Height [m]')
        
        # Phase text
        if t_idx < first_ts:
            phase = "Pre-formation"
            phase_color = 'gray'
        elif t_idx > last_ts:
            phase = "Post-dissipation"
            phase_color = 'gray'
        else:
            age = t_idx - first_ts
            phase = f"Age: {age} timesteps ({age * DT / 60:.1f} min)"
            phase_color = 'green'
        
        ax.set_title(f"Cloud {track_id} | t = {time_min:.1f} min | {phase}", 
                     fontsize=12, color=phase_color)
        
        # Fixed view following cloud horizontally
        ax.set_xlim(cx - VIEW_RANGE/2, cx + VIEW_RANGE/2)
        ax.set_ylim(cy - VIEW_RANGE/2, cy + VIEW_RANGE/2)
        ax.set_zlim(z_min, z_max)
        
        # Set view angle
        ax.view_init(elev=25, azim=45)
        
        return []
    
    # Create animation
    print("  Creating animation...")
    ani = animation.FuncAnimation(
        fig, update_frame, 
        frames=len(cloud_surfaces),
        interval=1000,  # 1 second per frame
        blit=False
    )
    
    # Save animation
    gif_file = f"{OUTPUT_FOLDER}/cloud_{track_id}_evolution.gif"
    ani.save(gif_file, writer='pillow', fps=1)
    print(f"  Saved animation: {gif_file}")
    
    # Save static images of first, middle, and last frames
    for frame_idx, label in [(0, 'start'), (len(cloud_surfaces)//2, 'middle'), (-1, 'end')]:
        update_frame(frame_idx)
        png_file = f"{OUTPUT_FOLDER}/cloud_{track_id}_{label}.png"
        plt.savefig(png_file, dpi=150, bbox_inches='tight')
        print(f"  Saved static: {png_file}")
    
    plt.close(fig)
    
    # Create base height evolution plot
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    
    times_min = [all_times[t] / 60 for t in valid_timesteps]
    ax2.plot(times_min, cloud_info['base_heights'], 'b-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Time [minutes]')
    ax2.set_ylabel('Cloud Base Height [m]')
    ax2.set_title(f'Cloud {track_id} - Base Height Evolution')
    ax2.grid(True, alpha=0.3)
    
    height_file = f"{OUTPUT_FOLDER}/cloud_{track_id}_base_height.png"
    plt.savefig(height_file, dpi=150, bbox_inches='tight')
    print(f"  Saved base height plot: {height_file}")
    plt.close(fig2)


def main():
    """Main entry point."""
    print("=" * 60)
    print("LBA Cloud 3D Visualizer")
    print("=" * 60)
    
    # Initialize MONC adapter
    print("\nInitializing MONC adapter...")
    adapter = MONCAdapter(MONC_CONFIG)
    
    # Find candidate clouds
    candidates = find_long_lived_clouds(MIN_LIFETIME_TIMESTEPS)
    
    if not candidates:
        print("\nNo clouds found meeting criteria!")
        return
    
    # Interactive selection
    while True:
        try:
            choice = input("\nEnter cloud index to visualize (or 'q' to quit): ")
            if choice.lower() == 'q':
                break
            
            idx = int(choice)
            if 0 <= idx < len(candidates):
                visualize_cloud_evolution(candidates[idx], adapter)
            else:
                print(f"Invalid index. Enter 0-{len(candidates)-1}")
        except ValueError:
            print("Please enter a valid number or 'q'")
        except KeyboardInterrupt:
            print("\nInterrupted.")
            break


if __name__ == "__main__":
    main()
