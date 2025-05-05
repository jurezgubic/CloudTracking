import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcolors
from netCDF4 import Dataset

def visualise_cloud_lifecycles(netcdf_file, output_file=None, max_tracks=30, min_valid_timesteps=3):
    """Create a visualisation showing cloud tracks over time (no merge arrows)."""
    with Dataset(netcdf_file, 'r') as ds:
        # Get valid tracks only
        valid_track = ds.variables['valid_track'][:]
        valid_indices = np.where(valid_track == 1)[0]
        
        if len(valid_indices) == 0:
            print("No valid tracks found in the dataset.")
            return
            
        # Determine which tracks have sufficient data
        size = ds.variables['size'][:]
        max_height = ds.variables['max_height'][:]
        
        # Calculate number of valid timesteps for each track
        track_data_points = []
        for i, idx in enumerate(valid_indices):
            # Count non-NaN entries in size data
            valid_timesteps = np.sum(~np.isnan(size[idx]))
            if valid_timesteps >= min_valid_timesteps:
                # Store track index, number of valid timesteps, and max cloud height
                track_max_height = np.nanmax(max_height[idx])
                track_data_points.append((idx, valid_timesteps, track_max_height))
        
        # Sort by number of valid timesteps (descending)
        track_data_points.sort(key=lambda x: x[1], reverse=True)
        
        if len(track_data_points) == 0:
            print(f"No valid tracks with at least {min_valid_timesteps} valid timesteps found.")
            return
        
        print(f"Found {len(track_data_points)} tracks with sufficient data out of {len(valid_indices)} valid tracks")
        
        # Select a balanced set of tracks: some long-lived, some medium, some short-lived
        selected_tracks = []
        if len(track_data_points) <= max_tracks:
            selected_tracks = [t[0] for t in track_data_points]
        else:
            # Divide into long, medium, and short lived
            third = len(track_data_points) // 3
            long_lived = track_data_points[:third]
            medium_lived = track_data_points[third:2*third]
            short_lived = track_data_points[2*third:]
            
            # Select proportionally from each group
            long_count = max(1, int(max_tracks * 0.5))  # 50% long-lived
            medium_count = max(1, int(max_tracks * 0.3))  # 30% medium-lived
            short_count = max_tracks - long_count - medium_count  # Remainder for short-lived
            
            # Ensure we don't try to select more than available
            long_count = min(long_count, len(long_lived))
            medium_count = min(medium_count, len(medium_lived))
            short_count = min(short_count, len(short_lived))
            
            # Select the tracks
            selected_tracks = [t[0] for t in long_lived[:long_count]]
            selected_tracks += [t[0] for t in medium_lived[:medium_count]]
            selected_tracks += [t[0] for t in short_lived[:short_count]]
        
        # Read data for selected tracks
        timesteps = ds.dimensions['time'].size
        
        # Create the visualisation
        plt.figure(figsize=(14, 8))
        
        # Use a colormap that provides good distinction between tracks
        cmap = plt.cm.tab20
        
        # Plot each cloud track
        for i, track_idx in enumerate(selected_tracks):
            # Find when this cloud exists
            cloud_exists = ~np.isnan(size[track_idx])
            cloud_timesteps = np.where(cloud_exists)[0]
            
            if len(cloud_timesteps) == 0:
                continue
            
            # Get color for this track
            color = cmap(i % 20)
            
            # Get marker sizes from cloud sizes
            track_sizes = size[track_idx, cloud_timesteps]
            
            # Normalize cloud sizes between 0 and 1
            min_size = np.nanmin(size)
            max_size = np.nanmax(size)
            normalized_sizes = (track_sizes - min_size) / (max_size - min_size)
            
            marker_sizes = np.clip(normalized_sizes * 180 + 20, 20, 200)  # Scale sizes for better visibility
            
            # Get y-position (use actual track index for better spacing)
            y_pos = i
            
            # Plot the track
            plt.scatter(cloud_timesteps, np.ones_like(cloud_timesteps) * y_pos, 
                      s=marker_sizes, c=[color]*len(cloud_timesteps), alpha=0.7, 
                      label=f"Track {track_idx}" if i < 10 else None)  # Only include first 10 in legend
            plt.plot(cloud_timesteps, np.ones_like(cloud_timesteps) * y_pos, 
                    '-', color=color, alpha=0.5)
            
            # Annotate with lifetime
            plt.text(cloud_timesteps[-1] + 0.3, y_pos, f"{len(cloud_timesteps)} steps", 
                    fontsize=8, va='center')
        
        # Add a grid
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Set labels and title
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Cloud Track', fontsize=12)
        plt.title('Cloud Lifecycles (Complete Lifecycle Tracks)', fontsize=16)
        
        # Set axis limits
        plt.xlim(-0.5, timesteps - 0.5)
        plt.ylim(-1, len(selected_tracks))
        
        # Set tick labels for y-axis (showing track indices)
        plt.yticks(range(len(selected_tracks)), 
                 [f"Track {track_idx}" for track_idx in selected_tracks])
        
        # Add legend for a subset of tracks to avoid clutter
        if len(selected_tracks) > 0:
            # Only show legend for first few clouds
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                plt.legend(handles=handles[:min(10, len(handles))], 
                          loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300)
            print(f"Cloud lifecycle visualisation saved to {output_file}")
        else:
            plt.show()