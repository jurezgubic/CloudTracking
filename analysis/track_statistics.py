import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

def compute_statistics(netcdf_file):
    """
    Compute comprehensive statistics about cloud tracks from NetCDF file.
    
    Only includes tracks marked as valid (valid_track=1), excluding all tracks
    with partial lifecycles (those present at timestep 0 or still active at the final timestep).
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
        
        # Count merge events
        merge_events = np.sum(merged_into > -1)
        stats['merge_events'] = merge_events
        
        # Count active clouds per timestep
        active_clouds_by_timestep = []
        timesteps = ds.dimensions['time'].size
        for t in range(timesteps):
            active_count = np.sum(~np.isnan(size[:, t]))
            active_clouds_by_timestep.append(active_count)
        stats['active_clouds_by_timestep'] = active_clouds_by_timestep
        
        # Compute age consistency (should always increment by 1)
        age_increments = []
        for i in range(n_tracks):
            if valid_track[i]:
                valid_ages = age[i][age[i] >= 0]
                if len(valid_ages) > 1:
                    # Check that age increments by 1 each timestep
                    increments = np.diff(valid_ages)
                    age_increments.extend(increments)
        
        stats['age_increments'] = age_increments
        if age_increments:
            stats['age_increment_mean'] = np.mean(age_increments)
            stats['age_increment_std'] = np.std(age_increments)
        
        # Compute track lifetimes and size statistics
        track_lifetimes = []
        size_at_birth = []
        size_at_death = []
        max_sizes = []
        
        # ToDo (optimisation): Replace these loops with vectorized NumPy operations
        for i in range(n_tracks):
            if valid_track[i]:
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
            
            # Filter for clouds with lifetime ≥ 3 timesteps
            long_lived_indices = [i for i, lifetime in enumerate(track_lifetimes) if lifetime >= 3]
            long_lived_lifetimes = [track_lifetimes[i] for i in long_lived_indices]
            
            if long_lived_lifetimes:
                stats['filtered_track_count'] = len(long_lived_lifetimes)
                stats['filtered_mean_lifetime'] = np.mean(long_lived_lifetimes)
                stats['filtered_median_lifetime'] = np.median(long_lived_lifetimes)
                stats['filtered_max_lifetime'] = np.max(long_lived_lifetimes)
                
                # Size statistics for filtered tracks
                filtered_birth_sizes = [size_at_birth[i] for i in long_lived_indices]
                filtered_death_sizes = [size_at_death[i] for i in long_lived_indices]
                filtered_max_sizes = [max_sizes[i] for i in long_lived_indices]
                
                stats['filtered_mean_size_at_birth'] = np.mean(filtered_birth_sizes)
                stats['filtered_mean_size_at_death'] = np.mean(filtered_death_sizes)
                stats['filtered_mean_max_size'] = np.mean(filtered_max_sizes)
                
                print(f"\nFiltered statistics (clouds with lifetime ≥ 3 timesteps):")
                print(f"  Count: {stats['filtered_track_count']} tracks ({stats['filtered_track_count']/len(track_lifetimes)*100:.1f}% of valid tracks)")
                print(f"  Mean lifetime: {stats['filtered_mean_lifetime']:.2f} timesteps")
                print(f"  Mean max size: {stats['filtered_mean_max_size']:.2f} grid points")
            else:
                print("\nNo clouds with lifetime ≥ 3 timesteps found")
                
        # NEW: Filter for robust clouds (≥3 timesteps AND minimum size ≥10 at all times)
        robust_cloud_indices = []
        for i in range(n_tracks):
            if valid_track[i]:
                # Check if the track has at least 3 timesteps with valid data
                track_ages = age[i][age[i] >= 0]
                if len(track_ages) >= 3:
                    # Check if all size values are >= 10
                    track_sizes = size[i][~np.isnan(size[i])]
                    if len(track_sizes) > 0 and np.all(track_sizes >= 10):
                        robust_cloud_indices.append(i)
        
        if robust_cloud_indices:
            # Calculate statistics for these robust clouds
            stats['robust_cloud_count'] = len(robust_cloud_indices)
            
            # Calculate active clouds by timestep for robust clouds
            robust_active_by_timestep = np.zeros(timesteps, dtype=int)
            for t in range(timesteps):
                # Only count clouds actually active at this timestep
                robust_active_count = sum(1 for idx in robust_cloud_indices 
                                         if t < size.shape[1] and not np.isnan(size[idx, t]))
                robust_active_by_timestep[t] = robust_active_count
            stats['robust_active_by_timestep'] = robust_active_by_timestep
            
            # Calculate lifetime statistics for robust clouds
            robust_indices_in_lifetimes = []
            for i in robust_cloud_indices:
                # Find this track's index in the track_lifetimes list
                for j, lifetime_idx in enumerate(range(n_tracks)):
                    if lifetime_idx == i and valid_track[lifetime_idx]:
                        robust_indices_in_lifetimes.append(j)
                        break
            
            robust_lifetimes = [track_lifetimes[i] for i in robust_indices_in_lifetimes if i < len(track_lifetimes)]
            
            # Directly calculate statistics from the cloud data
            if robust_lifetimes:
                stats['robust_mean_lifetime'] = np.mean(robust_lifetimes)
                print(f"DEBUG: Robust lifetimes: {robust_lifetimes[:10]}...")  # Check first 10 values
            
            if robust_lifetimes:
                stats['robust_mean_lifetime'] = np.mean(robust_lifetimes)
                stats['robust_median_lifetime'] = np.median(robust_lifetimes)
                stats['robust_max_lifetime'] = np.max(robust_lifetimes)
                
                # Get size statistics for robust clouds
                robust_size_indices = [i for i in range(len(size_at_birth)) if i in robust_indices_in_lifetimes]
                robust_birth_sizes = [size_at_birth[i] for i in robust_size_indices if i < len(size_at_birth)]
                robust_death_sizes = [size_at_death[i] for i in robust_size_indices if i < len(size_at_death)]
                robust_max_sizes = [max_sizes[i] for i in robust_size_indices if i < len(max_sizes)]
                
                if robust_birth_sizes:
                    stats['robust_mean_size_at_birth'] = np.mean(robust_birth_sizes)
                    stats['robust_mean_size_at_death'] = np.mean(robust_death_sizes)
                    stats['robust_mean_max_size'] = np.mean(robust_max_sizes)
                
                print(f"\nRobust cloud statistics (lifetime ≥ 3 AND size ≥ 10 at all times):")
                print(f"  Count: {stats['robust_cloud_count']} tracks " +
                      f"({stats['robust_cloud_count']/stats['valid_tracks']*100:.1f}% of valid tracks)")
                print(f"  Mean lifetime: {stats['robust_mean_lifetime']:.2f} timesteps")
                print(f"  Max lifetime: {stats['robust_max_lifetime']} timesteps")
                print(f"  Mean max size: {stats['robust_mean_max_size']:.2f} grid points")
        
        # Filter for robust clouds with relaxed criteria
        robust_cloud_indices = []
        for i in range(n_tracks):
            if valid_track[i]:
                # Check if the track has at least 3 timesteps with valid data
                track_ages = age[i][age[i] >= 0]
                if len(track_ages) >= 3:
                    # Option 1: Check if MOST sizes are >= 10 (at least 75%)
                    track_sizes = size[i][~np.isnan(size[i])]
                    if len(track_sizes) > 0 and np.mean(track_sizes >= 10) >= 0.75:
                        robust_cloud_indices.append(i)
                    
                    # Debug info - helps diagnose filtering issues
                    if len(track_sizes) > 0 and len(track_ages) >= 3:
                        min_size = np.min(track_sizes)
                        max_size = np.max(track_sizes)
                        if i < 10:  # Just print first 10 for debug
                            print(f"Cloud {i}: Lifetime={len(track_ages)}, Min size={min_size}, Max size={max_size}, Passing={np.mean(track_sizes >= 10) >= 0.75}")
    
    # Add debugging to see what's happening with robust clouds
    print(f"\nDEBUG: Found {len(robust_cloud_indices)} robust clouds")
    
    # Print statistics for first few robust clouds to verify criteria
    for i in range(min(5, len(robust_cloud_indices))):
        idx = robust_cloud_indices[i]
        track_sizes = size[idx][~np.isnan(size[idx])]
        track_ages = age[idx][age[idx] >= 0]
        print(f"  Cloud {idx}: lifetime={len(track_ages)} timesteps, "
              f"min_size={np.min(track_sizes):.1f}, "
              f"max_size={np.max(track_sizes):.1f}, "
              f"size≥10: {np.sum(track_sizes >= 10)}/{len(track_sizes)} timesteps")
    
    # Print per-timestep counts to verify data
    if 'robust_active_by_timestep' in stats:
        print(f"DEBUG: Robust active by timestep: {stats['robust_active_by_timestep']}")
    
    # ToDo (optimisation): Vectorized approach for counting active robust clouds
    robust_cloud_mask = np.zeros((len(robust_cloud_indices), timesteps), dtype=bool)
    for i, idx in enumerate(robust_cloud_indices):
        robust_cloud_mask[i] = ~np.isnan(size[idx, :])
    stats['robust_active_by_timestep'] = np.sum(robust_cloud_mask, axis=0)
    
    return stats

def visualise_statistics(stats, output_file=None):
    """Create visualisations of track statistics with added duration histogram and robust cloud analysis."""
    # Create 2x3 grid to accommodate all plots
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))
    
    # Plot 1: Active clouds by timestep
    timesteps = range(len(stats['active_clouds_by_timestep']))
    axs[0, 0].plot(timesteps, stats['active_clouds_by_timestep'], 'o-', color='blue')
    axs[0, 0].set_title('Active Clouds per Timestep')
    axs[0, 0].set_xlabel('Timestep')
    axs[0, 0].set_ylabel('Number of Active Clouds')
    axs[0, 0].grid(alpha=0.3)
    
    # Plot 2: Lifetime statistics
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
    size_labels = ['Birth Size', 'Death Size', 'Max Size']
    size_values = [
        stats['mean_size_at_birth'],
        stats['mean_size_at_death'],
        stats['mean_max_size']
    ]
    
    axs[0, 2].bar(size_labels, size_values, color=['lightblue', 'skyblue', 'royalblue'])
    axs[0, 2].set_title('Cloud Size Statistics')
    axs[0, 2].set_ylabel('Size (grid points)')
    
    # New histogram plot in the bottom row
    if 'lifetime_histogram' in stats:
        hist_values, hist_bins = stats['lifetime_histogram']
        axs[1, 0].bar(hist_bins[:-1], hist_values, width=0.8, align='edge')
        axs[1, 0].set_title('Cloud Lifetime Distribution')
        axs[1, 0].set_xlabel('Lifetime (timesteps)')
        axs[1, 0].set_ylabel('Number of Tracks')
        axs[1, 0].grid(alpha=0.3)
    
    # Add plot for filtered cloud statistics (lifetime ≥ 3)
    if 'filtered_mean_size_at_birth' in stats:
        filtered_size_labels = ['Birth Size', 'Death Size', 'Max Size']
        filtered_size_values = [
            stats['filtered_mean_size_at_birth'],
            stats['filtered_mean_size_at_death'],
            stats['filtered_mean_max_size']
        ]
        
        axs[1, 1].bar(filtered_size_labels, filtered_size_values, 
                     color=['lightgreen', 'green', 'darkgreen'])
        axs[1, 1].set_title(f'Size Statistics (Clouds ≥ 3 timesteps, n={stats["filtered_track_count"]})')
        axs[1, 1].set_ylabel('Size (grid points)')
        axs[1, 1].grid(alpha=0.3)
    else:
        axs[1, 1].set_visible(False)
    
    # NEW: Add plot for robust clouds
    if 'robust_active_by_timestep' in stats:
        timesteps = range(len(stats['robust_active_by_timestep']))
        axs[1, 2].plot(timesteps, stats['robust_active_by_timestep'], 'o-', color='red')
        axs[1, 2].set_title(f'Robust Clouds (≥3 timesteps, ≥10 size)\nn={stats["robust_cloud_count"]}')
        axs[1, 2].set_xlabel('Timestep')
        axs[1, 2].set_ylabel('Number of Active Clouds')
        axs[1, 2].grid(alpha=0.3)
    else:
        axs[1, 2].set_visible(False)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Statistics visualisation saved to {output_file}")
    else:
        plt.show()

