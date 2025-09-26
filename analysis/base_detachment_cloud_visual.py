import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from skimage import measure
import os
import sys

# Configuration parameters
lcl_tolerance = 50  # Tolerance for considering a cloud as starting at LCL (m)
detachment_threshold = 20  # Min cloud base height increase to be considered detached (m)
stable_timesteps = 2  # Min number of timesteps cloud base should be stable at LCL
min_lifetime_minutes = 5  # Minimum cloud lifetime (minutes)
timestep_duration_seconds = 60  # Duration between timesteps (seconds)
min_timesteps = int(min_lifetime_minutes * 60 / timestep_duration_seconds)
output_folder = 'detaching_cloud_visualizations'  # Folder to store visualizations
l_condition = 0.000001  # Liquid water content threshold (kg/kg)

# Path to original LES data - adjust as needed
base_file_path = '/Users/jure/PhD/coding/RICO_1hr/'
l_file_name = 'rico.l.nc'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def find_detaching_clouds():
    """Find clouds that start at LCL, stay stable for a few timesteps, then detach."""
    print("Finding clouds that start at LCL and later detach...")
    
    # Load the cloud tracking data
    filename = '../cloud_results.nc'
    with Dataset(filename, 'r') as nc:
        # Get variables
        valid_track = nc.variables['valid_track'][:]
        cloud_base_height = np.ma.filled(nc.variables['cloud_base_height'][:], np.nan)
        size = np.ma.filled(nc.variables['size'][:], np.nan)
        max_height = np.ma.filled(nc.variables['max_height'][:], np.nan)
        loc_x = np.ma.filled(nc.variables['location_x'][:], np.nan)
        loc_y = np.ma.filled(nc.variables['location_y'][:], np.nan)
        loc_z = np.ma.filled(nc.variables['location_z'][:], np.nan)
        
        # Get actual LCL height if available in the dataset
        lcl = 600  # Default value if not available
        if 'config_cloud_base_altitude' in nc.variables:
            lcl = float(nc.variables['config_cloud_base_altitude'][:])
            print(f"Using actual LCL height from dataset: {lcl}m")
        else:
            print(f"Using default LCL height: {lcl}m")
        
        # Find complete lifecycle clouds
        valid_cloud_indices = np.where(valid_track == 1)[0]
        print(f"Found {len(valid_cloud_indices)} clouds with complete lifecycles")
        
        detaching_clouds = []
        
        for idx in valid_cloud_indices:
            # Find timesteps where this cloud exists
            valid_timesteps = np.where(~np.isnan(cloud_base_height[idx]))[0]
            if len(valid_timesteps) < min_timesteps:
                continue
            
            first_timestep = valid_timesteps[0]
            last_timestep = valid_timesteps[-1]
            lifetime = last_timestep - first_timestep + 1
            
            # Extract cloud base height evolution
            base_heights = cloud_base_height[idx, valid_timesteps]
            
            # Check if the cloud starts near LCL
            if len(base_heights) < stable_timesteps + 2:  # Need enough timesteps to detect detachment
                continue
                
            initial_heights = base_heights[:stable_timesteps]
            
            # Check if initial height is near LCL and stable
            if not np.all(np.abs(initial_heights - lcl) <= lcl_tolerance):
                continue
                
            # Now check if the cloud base later detaches significantly
            later_heights = base_heights[stable_timesteps:]
            max_later_height = np.max(later_heights)
            
            # Calculate maximum detachment
            max_detachment = max_later_height - np.mean(initial_heights)
            
            if max_detachment >= detachment_threshold:
                # This cloud detaches! Record it
                # Find the timestep when detachment starts
                detachment_start = None
                for i, h in enumerate(base_heights[stable_timesteps:], start=stable_timesteps):
                    if h - np.mean(initial_heights) >= detachment_threshold/2:  # Halfway to threshold
                        detachment_start = valid_timesteps[i]
                        break
                
                # If we couldn't identify a clear detachment start, use a default
                if detachment_start is None:
                    detachment_start = first_timestep + stable_timesteps
                    
                detaching_clouds.append({
                    'track_id': idx,
                    'initial_height': float(np.mean(initial_heights)),
                    'max_detachment': float(max_detachment),
                    'first_timestep': int(first_timestep),
                    'detachment_timestep': int(detachment_start),
                    'last_timestep': int(last_timestep),
                    'lifetime': int(lifetime),
                    'max_height_ever': float(np.nanmax(max_height[idx])),
                    'max_size_ever': float(np.nanmax(size[idx])),
                    'base_height_evolution': base_heights.tolist(),
                    'valid_timesteps': valid_timesteps.tolist(),
                    'centroids': [(float(loc_x[idx, t]), float(loc_y[idx, t]), float(loc_z[idx, t])) 
                                for t in valid_timesteps]
                })
    
    print(f"Found {len(detaching_clouds)} clouds that start at LCL ({lcl}±{lcl_tolerance}m) " 
          f"and detach by at least {detachment_threshold}m later")
    
    # Sort clouds by detachment amount
    detaching_clouds.sort(key=lambda x: x['max_detachment'], reverse=True)
    
    # Display cloud information
    for i, cloud in enumerate(detaching_clouds):
        print(f"{i}. Cloud {cloud['track_id']}: starts at {cloud['initial_height']:.1f}m, "
              f"detaches by {cloud['max_detachment']:.1f}m at timestep {cloud['detachment_timestep']} "
              f"({cloud['lifetime']} total timesteps = {cloud['lifetime']*timestep_duration_seconds/60:.1f} minutes), "
              f"max height {cloud['max_height_ever']:.1f}m, max size {cloud['max_size_ever']:.1f}")
    
    return detaching_clouds

def extract_cloud_surface(l_data, xt, yt, zt, centroid, search_radius=2000):
    """
    Extract the cloud surface points around a centroid
    
    Args:
        l_data: 3D liquid water content data
        xt, yt, zt: Grid coordinates
        centroid: (x, y, z) position of cloud centroid
        search_radius: Maximum radius around centroid to search
        
    Returns:
        verts, faces: Vertices and faces of the cloud surface mesh
    """
    cx, cy, cz = centroid
    
    # Find grid indices corresponding to centroid
    x_idx = np.abs(xt - cx).argmin()
    y_idx = np.abs(yt - cy).argmin()
    z_idx = np.abs(zt - cz).argmin()
    
    # Define region of interest around centroid
    x_radius = np.ceil(search_radius / (xt[1] - xt[0])).astype(int)
    y_radius = np.ceil(search_radius / (yt[1] - yt[0])).astype(int)
    z_radius = np.ceil(search_radius / (zt[1] - zt[0])).astype(int)
    
    # Calculate bounds with domain boundary checks
    x_min = max(0, x_idx - x_radius)
    x_max = min(len(xt) - 1, x_idx + x_radius)
    y_min = max(0, y_idx - y_radius)
    y_max = min(len(yt) - 1, y_idx + y_radius)
    z_min = max(0, z_idx - z_radius)
    z_max = min(len(zt) - 1, z_idx + z_radius)
    
    # Extract subvolume
    subvolume = l_data[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
    
    # Create binary mask for cloud
    cloud_mask = subvolume >= l_condition
    
    # Skip if no cloud is found in this region
    if not np.any(cloud_mask):
        return None, None
    
    # Extract isosurface using marching cubes
    try:
        # Run marching cubes
        verts, faces, _, _ = measure.marching_cubes(
            cloud_mask.astype(float), 
            level=0.5,
            spacing=(
                zt[1] - zt[0],
                yt[1] - yt[0],
                xt[1] - xt[0]
            )
        )
        
        # Transform vertices to world coordinates
        verts[:, 0] += zt[z_min]
        verts[:, 1] += yt[y_min]
        verts[:, 2] += xt[x_min]
        
        return verts, faces
    
    except Exception as e:
        print(f"Error extracting isosurface: {e}")
        return None, None


def visualize_cloud_surface_evolution(cloud_info):
    """Create 3D visualization of cloud surface evolution that follows the cloud movement"""
    # First, load the grid coordinates
    with Dataset(f"{base_file_path}{l_file_name}", 'r') as dataset:
        xt = dataset.variables['xt'][:]
        yt = dataset.variables['yt'][:]
        zt = dataset.variables['zt'][:]
        total_available_timesteps = dataset.dimensions['time'].size
    
    # Prepare figure for visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get key timesteps
    cloud_lifecycle_start = cloud_info['first_timestep']
    cloud_detachment_timestep = cloud_info['detachment_timestep']
    cloud_lifecycle_end = cloud_info['last_timestep']
    valid_timesteps = cloud_info['valid_timesteps']
    
    # Create additional plot to show cloud base height evolution
    base_height_fig, base_height_ax = plt.subplots(figsize=(10, 5))
    base_heights = cloud_info['base_height_evolution']
    base_height_ax.plot(valid_timesteps, base_heights, 'b-', linewidth=2)
    base_height_ax.axvline(x=cloud_detachment_timestep, color='r', linestyle='--', 
                         label=f'Detachment starts (t={cloud_detachment_timestep})')
    base_height_ax.axhline(y=cloud_info['initial_height'], color='g', linestyle='--',
                         label=f'Initial height ({cloud_info["initial_height"]:.1f}m)')
    base_height_ax.axhline(y=cloud_info['initial_height'] + detachment_threshold, color='orange', 
                         linestyle='--', label=f'Detachment threshold (+{detachment_threshold}m)')
    base_height_ax.set_xlabel('Timestep')
    base_height_ax.set_ylabel('Cloud Base Height (m)')
    base_height_ax.set_title(f'Cloud {cloud_info["track_id"]} Base Height Evolution')
    base_height_ax.legend()
    base_height_ax.grid(True)
    
    # Save the base height evolution plot
    base_height_plot_file = f"{output_folder}/cloud_{cloud_info['track_id']}_base_height_evolution.png"
    base_height_fig.tight_layout()
    base_height_fig.savefig(base_height_plot_file)
    print(f"Saved base height evolution plot to {base_height_plot_file}")
    plt.close(base_height_fig)
    
    # Extend visualization range: 2 before, 1 after
    first_timestep = max(0, cloud_lifecycle_start - 2)
    last_timestep = min(total_available_timesteps - 1, cloud_lifecycle_end + 1)
    
    print(f"Extracting cloud surfaces for timesteps {first_timestep} to {last_timestep}...")
    
    # Store surface data for each timestep
    cloud_surfaces = []
    all_centroids = []
    
    # Create lookup dictionary for centroids
    centroid_map = {t: centroid for t, centroid in zip(valid_timesteps, cloud_info['centroids'])}
    
    for t in range(first_timestep, last_timestep + 1):
        print(f"Processing timestep {t}...")
        # Load liquid water content data for this timestep
        with Dataset(f"{base_file_path}{l_file_name}", 'r') as dataset:
            l_data = dataset.variables['l'][t, :, :, :]
        
        # Get centroid for this timestep
        if t in centroid_map:
            # Use exact centroid if available
            centroid = centroid_map[t]
        elif t < cloud_lifecycle_start and valid_timesteps:
            # For timesteps before cloud appears, use first centroid position
            centroid = centroid_map[valid_timesteps[0]]
        elif t > cloud_lifecycle_end and valid_timesteps:
            # For timesteps after cloud disappears, use last centroid position
            centroid = centroid_map[valid_timesteps[-1]]
        else:
            # Fallback if somehow we don't have centroids
            centroid = (0, 0, 0)
            
        all_centroids.append(centroid)
        
        # Extract cloud surface
        verts, faces = extract_cloud_surface(l_data, xt, yt, zt, centroid)
        
        if verts is not None and faces is not None:
            cloud_surfaces.append((t, verts, faces, centroid))
            print(f"  Found cloud surface with {len(verts)} vertices")
        else:
            # If no cloud is found, still keep track of the empty timestep
            cloud_surfaces.append((t, None, None, centroid))
            print(f"  No cloud surface found at timestep {t}")
    
    if not any(surf[1] is not None for surf in cloud_surfaces):
        print("No cloud surfaces found for visualization")
        return
    
    # Calculate fixed view height range (z) for consistent vertical perspective
    valid_surfaces = [surf for surf in cloud_surfaces if surf[1] is not None]
    if valid_surfaces:
        all_verts = np.vstack([surf[1] for surf in valid_surfaces])
        z_min = np.min(all_verts[:, 0]) - 200  # Add padding
        z_max = np.max(all_verts[:, 0]) + 200
    else:
        # Default height range if no valid surfaces
        z_min = 0
        z_max = 3000
    
    # Function to update plot with a specific timestep's surface
    def update_surface(timestep_idx):
        ax.clear()
        
        # Get surface data for this timestep
        t, verts, faces, centroid = cloud_surfaces[timestep_idx]
        
        # Calculate view center based on centroid position
        center_x, center_y = centroid[0], centroid[1]
        
        # Set window size around the cloud
        view_range = 2000  # meters
        
        if verts is not None and faces is not None:
            # Determine color based on whether detachment has occurred
            if t >= cloud_detachment_timestep:
                cmap = 'plasma'  # Use different colormap for detached phase
            else:
                cmap = 'viridis'  # Use standard colormap for initial phase
                
            # Plot the surface mesh
            mesh = ax.plot_trisurf(
                verts[:, 2], verts[:, 1], verts[:, 0],  # x, y, z order for plot_trisurf
                triangles=faces,
                cmap=cmap,
                alpha=0.8
            )
        else:
            # Create empty mesh if no cloud is found
            mesh = None
        
        # Always mark the centroid position
        ax.scatter(
            centroid[0], centroid[1], centroid[2],
            color='red', s=200, marker='o', edgecolor='black'
        )
        
        # Add a horizontal plane at the initial cloud base height
        x_range = np.array([center_x - view_range/2, center_x + view_range/2])
        y_range = np.array([center_y - view_range/2, center_y + view_range/2])
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.ones_like(xx) * cloud_info['initial_height']
        ax.plot_surface(xx, yy, zz, color='green', alpha=0.2)
        
        # Add indicators for cloud lifecycle phase
        phase_text = ""
        if t < cloud_lifecycle_start:
            phase_text = "Pre-formation"
        elif t > cloud_lifecycle_end:
            phase_text = "Post-dissipation"
        elif t < cloud_detachment_timestep:
            phase_text = f"Initial phase (Age: {t - cloud_lifecycle_start})"
        else:
            phase_text = f"Detached phase (Age: {t - cloud_lifecycle_start})"
        
        # Set axis labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"Cloud {cloud_info['track_id']} Surface at Timestep {t}")
        
        # Set consistent moving view that follows the cloud horizontally
        ax.set_xlim(center_x - view_range/2, center_x + view_range/2)
        ax.set_ylim(center_y - view_range/2, center_y + view_range/2)
        ax.set_zlim(z_min, z_max)  # Keep vertical scale constant
        
        # Add timestep and phase indicators
        ax.text2D(0.05, 0.95, f"Timestep: {t}", transform=ax.transAxes)
        
        # Color code the phase text
        color = 'red' if t < cloud_lifecycle_start or t > cloud_lifecycle_end else \
                'green' if t < cloud_detachment_timestep else 'orange'
        ax.text2D(0.05, 0.90, phase_text, transform=ax.transAxes, color=color)
        
        # Show current cloud base height if available
        if t in centroid_map:
            time_idx = valid_timesteps.index(t)
            current_height = cloud_info['base_height_evolution'][time_idx]
            ax.text2D(0.05, 0.85, f"Base Height: {current_height:.1f}m", transform=ax.transAxes)
            
            # Add a small marker at current cloud base height
            ax.scatter(centroid[0], centroid[1], current_height, 
                      color='cyan', s=100, marker='s', edgecolor='black')
        
        return [mesh] if mesh else []
    
    # Create animation
    print("Creating animation...")
    ani = animation.FuncAnimation(
        fig, 
        update_surface, 
        frames=len(cloud_surfaces),
        interval=1000,  # milliseconds
        blit=False
    )
    
    # Save animation
    animation_file = f"{output_folder}/cloud_{cloud_info['track_id']}_detachment_evolution.gif"
    ani.save(animation_file, writer='pillow', fps=1)
    print(f"Saved animation to {animation_file}")
    
    # Calculate indices for key timesteps in our visualization sequence
    pre_detach_offset = cloud_detachment_timestep - first_timestep - 1
    pre_detach_offset = max(0, min(pre_detach_offset, len(cloud_surfaces)-1))
    
    post_detach_offset = cloud_detachment_timestep - first_timestep + 2
    post_detach_offset = max(0, min(post_detach_offset, len(cloud_surfaces)-1))
    
    # Save static images of key phases
    static_file1 = f"{output_folder}/cloud_{cloud_info['track_id']}_pre_detachment.png"
    update_surface(pre_detach_offset)
    plt.savefig(static_file1)
    print(f"Saved pre-detachment image to {static_file1}")
    
    static_file2 = f"{output_folder}/cloud_{cloud_info['track_id']}_post_detachment.png"
    update_surface(post_detach_offset)
    plt.savefig(static_file2)
    print(f"Saved post-detachment image to {static_file2}")
    
    plt.show()


# Main execution
if __name__ == "__main__":
    # Find clouds that meet detachment criteria
    detaching_clouds = find_detaching_clouds()
    
    if detaching_clouds:
        while True:
            try:
                choice = input("\nEnter the index of the detaching cloud to visualize (or 'q' to quit): ")
                if choice.lower() == 'q':
                    break
                    
                choice_idx = int(choice)
                if 0 <= choice_idx < len(detaching_clouds):
                    selected_cloud = detaching_clouds[choice_idx]
                    print(f"\nVisualizing Detaching Cloud {selected_cloud['track_id']}...")
                    visualize_cloud_surface_evolution(selected_cloud)
                else:
                    print(f"Invalid index. Please enter a number between 0 and {len(detaching_clouds) - 1}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    else:
        print("No clouds meeting the detachment criteria were found.")