import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset

def compute_statistics(netcdf_file):
    """Compute comprehensive statistics about cloud tracks from NetCDF file."""
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
                ages = age[i][age[i] >= 0]
                if len(ages) > 1:
                    increments = ages[1:] - ages[:-1]
                    age_increments.extend(increments)
        
        stats['consistent_ages'] = np.all(np.array(age_increments) == 1) if age_increments else False
        stats['age_increment_issues'] = np.sum(np.array(age_increments) != 1) if age_increments else 0
        
        # Compute lifetimes
        track_lifetimes = []
        for i in range(n_tracks):
            if valid_track[i]:
                valid_ages = age[i][age[i] >= 0]
                if len(valid_ages) > 0:
                    track_lifetimes.append(np.max(valid_ages) + 1)  # +1 because age starts at 0
        
        stats['mean_lifetime'] = np.mean(track_lifetimes) if track_lifetimes else 0
        stats['median_lifetime'] = np.median(track_lifetimes) if track_lifetimes else 0
        stats['max_lifetime'] = max(track_lifetimes) if track_lifetimes else 0
        
        # Size evolution
        size_at_birth = []
        size_at_death = []
        max_sizes = []
        
        for i in range(n_tracks):
            if valid_track[i]:
                # Find indices where cloud exists
                valid_indices = np.where(~np.isnan(size[i, :]))[0]
                if len(valid_indices) > 0:
                    birth_idx = valid_indices[0]
                    death_idx = valid_indices[-1]
                    
                    size_at_birth.append(size[i, birth_idx])
                    size_at_death.append(size[i, death_idx])
                    max_sizes.append(np.max(size[i, valid_indices]))
        
        stats['mean_size_at_birth'] = np.mean(size_at_birth) if size_at_birth else 0
        stats['mean_size_at_death'] = np.mean(size_at_death) if size_at_death else 0
        stats['mean_max_size'] = np.mean(max_sizes) if max_sizes else 0
        
    return stats

def visualise_statistics(stats, output_file=None):
    """Create visualisations of track statistics without the track counts plot."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Active clouds by timestep
    timesteps = range(len(stats['active_clouds_by_timestep']))
    axs[0].plot(timesteps, stats['active_clouds_by_timestep'], 'o-', color='blue')
    axs[0].set_title('Active Clouds per Timestep')
    axs[0].set_xlabel('Timestep')
    axs[0].set_ylabel('Number of Active Clouds')
    axs[0].grid(alpha=0.3)
    
    # Plot 2: Lifetime statistics
    lifetime_labels = ['Mean', 'Median', 'Max']
    lifetime_values = [
        stats['mean_lifetime'], 
        stats['median_lifetime'],
        stats['max_lifetime']
    ]
    
    axs[1].bar(lifetime_labels, lifetime_values, color=['purple', 'magenta', 'pink'])
    axs[1].set_title('Cloud Lifetime Statistics')
    axs[1].set_ylabel('Timesteps')
    
    # Plot 3: Size statistics
    size_labels = ['Birth Size', 'Death Size', 'Max Size']
    size_values = [
        stats['mean_size_at_birth'],
        stats['mean_size_at_death'],
        stats['mean_max_size']
    ]
    
    axs[2].bar(size_labels, size_values, color=['lightblue', 'skyblue', 'royalblue'])
    axs[2].set_title('Cloud Size Statistics')
    axs[2].set_ylabel('Size (grid points)')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
        print(f"Statistics visualisation saved to {output_file}")
    else:
        plt.show()