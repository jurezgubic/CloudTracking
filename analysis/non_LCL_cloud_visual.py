import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.tri import Triangulation
from skimage import measure
import os
import sys

# Configuration parameters
height_threshold = 750  # Minimum height for cloud initiation (m)
min_lifetime_minutes = 5  # Minimum cloud lifetime (minutes)
timestep_duration_seconds = 60  # Duration between timesteps (seconds)
min_timesteps = int(min_lifetime_minutes * 60 / timestep_duration_seconds)
output_folder = 'cloud_surface_visualizations'  # Folder to store visualizations
l_condition = 0.00001  # Liquid water content threshold (kg/kg)

# Path to original LES data
base_file_path = '/Users/jure/PhD/coding/RICO_1hr/'
l_file_name = 'rico.l.nc'

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

def find_high_initiated_clouds():
    """Find clouds that initiate above the height threshold with minimum lifetime."""
    print("Finding clouds initiated above cloud base...")
    # Load the cloud tracking data
    filename = '../cloud_results_0.00001_12min.nc'
    with Dataset(filename, 'r') as nc:
        # Get variables
        valid_track = nc.variables['valid_track'][:]
        cloud_base_height = nc.variables['cloud_base_height'][:]
        size = nc.variables['size'][:]
        max_height = nc.variables['max_height'][:]
        age = nc.variables['age'][:]
        loc_x = nc.variables['location_x'][:]
        loc_y = nc.variables['location_y'][:]
        loc_z = nc.variables['location_z'][:]
        
        # Find complete lifecycle clouds
        valid_cloud_indices = np.where(valid_track == 1)[0]
        print(f"Found {len(valid_cloud_indices)} clouds with complete lifecycles")
        
        high_initiated_clouds = []
        
        for idx in valid_cloud_indices:
            # Find first and last timesteps where this cloud exists
            valid_timesteps = np.where(~np.isnan(cloud_base_height[idx]))[0]
            if len(valid_timesteps) == 0:
                continue
            
            first_timestep = valid_timesteps[0]
            last_timestep = valid_timesteps[-1]
            lifetime = last_timestep - first_timestep + 1
            
            # Check initial height and lifetime
            initial_height = cloud_base_height[idx, first_timestep]
            
            if initial_height > height_threshold and lifetime >= min_timesteps:
                high_initiated_clouds.append({
                    'track_id': idx,
                    'initial_height': initial_height,
                    'first_timestep': first_timestep,
                    'last_timestep': last_timestep,
                    'lifetime': lifetime,
                    'max_height_ever': np.nanmax(max_height[idx]),
                    'max_size_ever': np.nanmax(size[idx]),
                    'centroids': [(loc_x[idx, t], loc_y[idx, t], loc_z[idx, t]) 
                                for t in range(first_timestep, last_timestep+1)]
                })
    
    print(f"Found {len(high_initiated_clouds)} clouds that initiated above {height_threshold}m " 
          f"and lived at least {min_lifetime_minutes} minutes")
    
    # Display cloud information
    for i, cloud in enumerate(high_initiated_clouds):
        print(f"{i}. Cloud {cloud['track_id']}: initiated at {cloud['initial_height']:.1f}m, "
              f"timesteps {cloud['first_timestep']}-{cloud['last_timestep']} "
              f"({cloud['lifetime']} timesteps = {cloud['lifetime']*timestep_duration_seconds/60:.1f} minutes), "
              f"max height {cloud['max_height_ever']:.1f}m, max size {cloud['max_size_ever']:.1f}")
    
    return high_initiated_clouds

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
        # Create coordinate arrays for the subvolume
        z_sub, y_sub, x_sub = np.meshgrid(
            zt[z_min:z_max+1],
            yt[y_min:y_max+1],
            xt[x_min:x_max+1],
            indexing='ij'
        )
        
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
    
    # List to store surface mesh data for each timestep
    cloud_surfaces = []
    
    # Extend visualization range: 2 before, 1 after
    first_timestep = max(0, cloud_info['first_timestep'] - 2)
    last_timestep = min(total_available_timesteps - 1, cloud_info['last_timestep'] + 1)
    
    print(f"Extracting cloud surfaces for extended timesteps {first_timestep} to {last_timestep}...")
    
    # Store centroids for all timesteps including empty ones
    all_centroids = []
    cloud_lifecycle_start = cloud_info['first_timestep']
    cloud_lifecycle_end = cloud_info['last_timestep']
    
    for t in range(first_timestep, last_timestep + 1):
        print(f"Processing timestep {t}...")
        # Load liquid water content data for this timestep
        with Dataset(f"{base_file_path}{l_file_name}", 'r') as dataset:
            l_data = dataset.variables['l'][t, :, :, :]
        
        # Get centroid for this timestep if within cloud lifecycle, otherwise use previous/first centroid
        if cloud_lifecycle_start <= t <= cloud_lifecycle_end:
            centroid = cloud_info['centroids'][t - cloud_lifecycle_start]
        elif t < cloud_lifecycle_start and all_centroids:
            # For timesteps before cloud appears, use first centroid position
            centroid = cloud_info['centroids'][0]
        elif t < cloud_lifecycle_start:
            # If this is the very first timestep and cloud hasn't appeared yet
            centroid = cloud_info['centroids'][0]  # Use first known position
        else:
            # For timesteps after cloud disappears, use last centroid position
            centroid = cloud_info['centroids'][-1]
            
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
        
        # Set window size around the cloud (fixed size view that moves with cloud)
        view_range = 2000  # meters
        
        if verts is not None and faces is not None:
            # Plot the surface mesh
            mesh = ax.plot_trisurf(
                verts[:, 2], verts[:, 1], verts[:, 0],  # x, y, z order for plot_trisurf
                triangles=faces,
                cmap='viridis',
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
        
        # Add indicators for cloud lifecycle phase
        phase_text = ""
        if t < cloud_lifecycle_start:
            phase_text = "Pre-formation"
        elif t > cloud_lifecycle_end:
            phase_text = "Post-dissipation"
        else:
            phase_text = f"Active (Age: {t - cloud_lifecycle_start})"
        
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
        ax.text2D(0.05, 0.90, phase_text, transform=ax.transAxes, 
                 color='green' if cloud_lifecycle_start <= t <= cloud_lifecycle_end else 'red')
        
        return [mesh] if mesh else []
    
    # Create animation
    print("Creating animation...")
    ani = animation.FuncAnimation(
        fig, 
        update_surface, 
        frames=len(cloud_surfaces),
        interval=1500,  # milliseconds
        blit=False
    )
    
    # Save animation
    animation_file = f"{output_folder}/cloud_{cloud_info['track_id']}_surface_evolution.gif"
    ani.save(animation_file, writer='pillow', fps=1)
    print(f"Saved animation to {animation_file}")
    
    # Display the first timestep
    update_surface(0)
    plt.tight_layout()
    
    # Also save a static image of the first active timestep
    active_idx = cloud_lifecycle_start - first_timestep
    static_file = f"{output_folder}/cloud_{cloud_info['track_id']}_surface_timestep_{cloud_lifecycle_start}.png"
    update_surface(active_idx)
    plt.savefig(static_file)
    print(f"Saved static image to {static_file}")
    
    plt.show()



# Main execution
if __name__ == "__main__":
    # Find clouds that meet criteria
    high_initiated_clouds = find_high_initiated_clouds()
    
    if high_initiated_clouds:
        while True:
            try:
                choice = input("\nEnter the index of the cloud to visualize (or 'q' to quit): ")
                if choice.lower() == 'q':
                    break
                    
                choice_idx = int(choice)
                if 0 <= choice_idx < len(high_initiated_clouds):
                    selected_cloud = high_initiated_clouds[choice_idx]
                    print(f"\nVisualizing Cloud {selected_cloud['track_id']}...")
                    visualize_cloud_surface_evolution(selected_cloud)
                else:
                    print(f"Invalid index. Please enter a number between 0 and {len(high_initiated_clouds) - 1}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    else:
        print("No clouds meeting the criteria were found.")