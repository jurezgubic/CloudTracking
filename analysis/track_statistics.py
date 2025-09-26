import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

def compute_statistics(netcdf_file, min_timesteps=3, min_size=10):
    """
    Compute comprehensive statistics about cloud tracks from NetCDF file.
    
    Only includes tracks that meet all criteria:
    1. Complete lifecycle (valid_track=1)
    2. Minimum lifetime of min_timesteps
    3. At least one timestep with size >= min_size
    """
    stats = {}
    
    with Dataset(netcdf_file, 'r') as ds:
        # Get dimensions and key variables
        n_tracks = ds.dimensions['track'].size
        valid_track = ds.variables['valid_track'][:]
        size = ds.variables['size'][:]
        age = ds.variables['age'][:]
        merged_into = ds.variables['merged_into'][:]
        
        # Basic counts
        stats['total_tracks'] = n_tracks
        stats['valid_tracks'] = np.sum(valid_track == 1)
        stats['partial_tracks'] = n_tracks - stats['valid_tracks']
        
        # Identify filtered tracks meeting all criteria
        filtered_track_indices = []
        for i in range(n_tracks):
            if valid_track[i]:  # Only complete lifecycle clouds
                # Check lifetime
                track_ages = age[i][age[i] >= 0]
                track_sizes = size[i][~np.isnan(size[i])]
                
                if len(track_ages) >= min_timesteps and len(track_sizes) > 0:
                    # Check if there's at least one timestep where size >= min_size
                    if np.any(track_sizes >= min_size):
                        filtered_track_indices.append(i)
        
        stats['filtered_tracks'] = len(filtered_track_indices)
        print(f"Found {stats['filtered_tracks']} tracks meeting all criteria out of {stats['valid_tracks']} valid tracks")
        
        # Only continue if we have tracks meeting criteria
        if not filtered_track_indices:
            print("No tracks meet the filtering criteria. Cannot compute further statistics.")
            return stats
            
        # Count merge events among filtered tracks
        merge_events = np.sum(merged_into[filtered_track_indices] > -1)
        stats['merge_events'] = merge_events
        
        # Count active filtered clouds per timestep
        filtered_clouds_by_timestep = []
        timesteps = ds.dimensions['time'].size
        for t in range(timesteps):
            active_count = np.sum(~np.isnan(size[filtered_track_indices, t]))
            filtered_clouds_by_timestep.append(active_count)
        stats['active_clouds_by_timestep'] = filtered_clouds_by_timestep
        
        # Compute track lifetimes and size statistics for filtered clouds only
        track_lifetimes = []
        size_at_birth = []
        size_at_death = []
        max_sizes = []
        
        for i in filtered_track_indices:
            valid_ages = age[i][age[i] >= 0]
            valid_sizes = size[i][~np.isnan(size[i])]
            
            if len(valid_ages) > 0 and len(valid_sizes) > 0:
                lifetime = np.max(valid_ages) + 1  # Add 1 since age starts at 0
                track_lifetimes.append(lifetime)
                size_at_birth.append(valid_sizes[0])
                size_at_death.append(valid_sizes[-1])
                max_sizes.append(np.max(valid_sizes))
        
        # Track lifetime statistics
        if track_lifetimes:
            stats['track_lifetimes'] = track_lifetimes
            stats['mean_lifetime'] = np.mean(track_lifetimes)
            stats['median_lifetime'] = np.median(track_lifetimes)
            stats['max_lifetime'] = np.max(track_lifetimes)
            
            # Size statistics
            stats['mean_size_at_birth'] = np.mean(size_at_birth)
            stats['mean_size_at_death'] = np.mean(size_at_death)
            stats['mean_max_size'] = np.mean(max_sizes)
            
            # Add histogram data for track durations
            stats['lifetime_histogram'] = np.histogram(
                track_lifetimes, 
                bins=range(1, max(track_lifetimes) + 2)
            )
            
            # Count tracks by duration
            unique_durations, duration_counts = np.unique(track_lifetimes, return_counts=True)
            stats['tracks_by_duration'] = {int(dur): int(count) for dur, count in zip(unique_durations, duration_counts)}
            
            print("\nTrack duration distribution:")
            for duration, count in sorted(stats['tracks_by_duration'].items()):
                print(f"  {duration} timesteps: {count} tracks ({count/len(track_lifetimes)*100:.1f}%)")
    
    return stats

def visualise_statistics(stats, output_file=None):
    """Create visualisations of track statistics for filtered clouds only."""
    # Create 2x2 grid (simpler layout)
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Active clouds by timestep
    if 'active_clouds_by_timestep' in stats:
        timesteps = range(len(stats['active_clouds_by_timestep']))
        axs[0, 0].plot(timesteps, stats['active_clouds_by_timestep'], 'o-', color='blue')
        axs[0, 0].set_title(f'Active Filtered Clouds per Timestep (n={stats["filtered_tracks"]})')
        axs[0, 0].set_xlabel('Timestep')
        axs[0, 0].set_ylabel('Number of Active Clouds')
        axs[0, 0].grid(alpha=0.3)
    
    # Plot 2: Lifetime statistics
    if 'mean_lifetime' in stats:
        lifetime_labels = ['Mean', 'Median', 'Max']
        lifetime_values = [
            stats['mean_lifetime'], 
            stats['median_lifetime'],
            stats['max_lifetime']
        ]
        
        axs[0, 1].bar(lifetime_labels, lifetime_values, color=['purple', 'magenta', 'pink'])
        axs[0, 1].set_title('Cloud Lifetime Statistics')
        axs[0, 1].set_ylabel('Timesteps')
    
    # Plot 3: Size statistics
    if 'mean_size_at_birth' in stats:
        size_labels = ['Birth Size', 'Death Size', 'Max Size']
        size_values = [
            stats['mean_size_at_birth'],
            stats['mean_size_at_death'],
            stats['mean_max_size']
        ]
        
        axs[1, 0].bar(size_labels, size_values, color=['lightblue', 'skyblue', 'royalblue'])
        axs[1, 0].set_title('Cloud Size Statistics')
        axs[1, 0].set_ylabel('Size (grid points)')
    
    # Plot 4: Duration histogram
    if 'lifetime_histogram' in stats:
        hist_values, hist_bins = stats['lifetime_histogram']
        axs[1, 1].bar(hist_bins[:-1], hist_values, width=0.8, align='edge', color='green')
        axs[1, 1].set_title('Cloud Lifetime Distribution')
        axs[1, 1].set_xlabel('Lifetime (timesteps)')
        axs[1, 1].set_ylabel('Number of Tracks')
        axs[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Statistics visualisation saved to {output_file}")
    else:
        plt.show()

