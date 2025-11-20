import argparse, os, gc, time
from memory_profiler import profile
from src.data_management import load_cloud_field_from_file, calculate_mean_velocities
from src.netcdf_writer import write_cloud_tracks_to_netcdf
from lib.cloudtracker import CloudTracker
import src.data_management as data_management
import numpy as np
from netCDF4 import Dataset

# Warning!
# User needs to modify: base_file_path, output_netcdf_path, file_names, total_timesteps, and config

# --- Start of user modifiable parameters ---

# Set file paths and parameters
base_file_path = '/Users/jure/PhD/coding/RICO_1hr/'
output_netcdf_path = 'cloud_results.nc'

# paths to the LES data files
file_name = {
    'l': 'rico.l.nc',  # Liquid water mixing ratio
    'u': 'rico.u.nc',  # u wind
    'v': 'rico.v.nc',  # v wind
    'w': 'rico.w.nc',  # w wind (vertical velocity)
    'p': 'rico.p.nc',  # Pressure
    't': 'rico.t.nc',  # Liquid Water Potential temperature
    'q': 'rico.q.nc',  # Total water mixing ratio
}

# Processing Options
total_timesteps = 3 # Number of timesteps to process

# Cloud Definition and Tracking Configuration
config = {
    # Cloud identification
    'min_size': 10,              # Minimum number of points for a cloud to be considered a cloud
    'l_condition': 0.01,      # kg/kg. Minimum liquid water content for a point to be a cloud.
    'w_condition': 0.0,          # m/s. Minimum vertical velocity for a point to be part of a cloud.
    'w_switch': False,           # If True, apply the 'w_condition' threshold.
    
    # Simulation parameters
    'timestep_duration': 60,     # Seconds. Time between timesteps.
    'horizontal_resolution': 25.0, # m. Grid resolution from the simulation namelist.

    # Physics-based Adjustments
    'switch_wind_drift': True,   # If True, consider the mean horizontal motion in tracking.
    'switch_vertical_drift': True, # If True, consider the mean vertical motion in tracking.
    'cloud_base_altitude': 650,  # m. Estimated cloud base altitude for certain calculations

    # Matching parameters
    'distance_threshold': 0,     # Max distance between merging clouds across a periodic boundary.
    'min_h_match_factor': 2.5,   # Minimum horizontal match factor. Min distance =  'min_h_match_factor' * 'horizontal_resolution'
    'min_v_match_factor': 2.5,   # Minimum vertical match factor. Min distance =  'min_v_match_factor' * 'horizontal_resolution'
    'match_safety_factor_dynamic': 2.0,  # Dynamic safety factor for matching clouds based on velocities. 
    'bounding_box_safety_factor': 1.2, # Safety factor for pre-filtering potential matches (using centroids).
    'max_expected_cloud_speed': 15.0,  # m/s. An estimate to constrain the search space for matching.
    'use_pre_filtering': True,   # If True, use a pre-filtering step to find potential matches (speed up matching).
    'switch_prefilter_fallback': False, # If True: when no pre-filter candidates are found fallback to full-domain search.
    'min_surface_overlap_points': 10,  # Require at least this many overlapping surface points for a match
    'match_shell_layers': 3,       # Number of cloud shell layers to include for matching (1 = surface points only)

    # Visualisation (somewhat deprecated)
    'plot_switch': False,        # If True, plot the cloud field at each timestep.

    # Parameters for cloud base diagnosisq
    'base_scan_levels': 3,            # Number of levels to scan upward for diagnosed base from lowest cloud level
    'base_increase_threshold': 1.5,   # Factor required to increase base radius from lowest cloud level (1.5 = 50%)

    # NIP (Neighbour Interaction Potential) parameters
    'nip_gamma': 0.3,           # Kinematic boost coefficient
    'nip_f': 3.0,               # Radius multiplier for neighbour search per level
    'nip_Lh_min': 100.0,        # Min horizontal scale Lh [m]
    'nip_Lh_max': 2000.0,       # Max horizontal scale Lh [m]
    'nip_T_min': 60.0,          # Min temporal memory scale [s]
    'nip_T_max': 6000.0,        # Max temporal memory scale [s]

    # Environment ring (per-cloud surroundings) parameters
    'env_ring_max_distance': 3,   # Max Manhattan ring distance D around cloud edge (2D)
    'env_periodic_rings': True,   # Respect periodic boundaries when forming rings
}
# --- End of user modifiable parameters ---



# Function to calculate mean_w
def calculate_mean_vertical_velocity(file_path, file_names, timestep):
    """Calculate mean vertical velocity for each z-level at the given timestep."""
    w_data = data_management.load_w_field(file_path, file_names['w'], timestep)
    mean_w = np.mean(w_data, axis=(1, 2))  # Average over y and x dimensions
    return mean_w

def resolve_cloud_base_altitude(config, z_levels):
    """
    Resolve user-provided cloud_base_altitude (meters) to nearest model level.
    Stores:
      config['cloud_base_level_index']
      config['cloud_base_altitude_resolved']
    """
    target = config.get('cloud_base_altitude', None)
    if target is None:
        raise ValueError("config must contain 'cloud_base_altitude'")
    z_arr = np.asarray(z_levels, dtype=float)
    if z_arr.ndim != 1 or z_arr.size == 0:
        raise ValueError("z_levels must be a 1D non-empty array")
    idx = int(np.abs(z_arr - target).argmin())
    resolved = float(z_arr[idx])
    config['cloud_base_level_index'] = idx
    config['cloud_base_altitude_resolved'] = resolved
    print(f"Cloud base requested at {target}m. Nearest resolved (and used) height is {resolved}m.")

# @profile
def process_clouds(cloud_tracker):
    """Process clouds and mark partial lifecycle clouds as tainted."""
    tainted_count = 0
    base_resolved = False
    
    for timestep in range(total_timesteps):
        start = time.time()
        print("-"*50)
        print(f"Processing timestep {timestep+1} of {total_timesteps}")

        cloud_field = data_management.load_cloud_field_from_file(
            base_file_path, file_name, timestep, config
        )

        # Resolve base altitude once (after first load when zt is known)
        if not base_resolved:
            resolve_cloud_base_altitude(config, cloud_field.zt)
            base_resolved = True

        cloud_tracker.update_tracks(cloud_field, cloud_field.zt, cloud_field.xt, cloud_field.yt)

        # Handle Partial Lifecycles (Tainted tracks)
        # To analyse a cloud's full lifecycle, we must see its birth and death within the simulation.
        # Clouds present at the start or end of the simulation, or those that merge with these
        # clouds, have incomplete lifecycles and are marked as tainted.

        # Handle clouds present at the first timestep.
        if timestep == 0:
            timestep_0_ids = set(cloud_tracker.cloud_tracks.keys())
            cloud_tracker.tainted_tracks.update(timestep_0_ids)
            tainted_count += len(timestep_0_ids)
            print(f"Marked {len(timestep_0_ids)} partial lifetime clouds from first timestep")
        else:
            # Check for clouds that merged with tainted tracks
            newly_tainted = []
            for track_id, track in cloud_tracker.cloud_tracks.items():
                if track_id in cloud_tracker.tainted_tracks:
                    continue
                    
                for cloud in track:
                    if hasattr(cloud, 'merged_into') and cloud.merged_into in cloud_tracker.tainted_tracks:
                        newly_tainted.append(track_id)
                        break
            
            # Mark as tainted instead of deleting
            cloud_tracker.tainted_tracks.update(newly_tainted)
            tainted_count += len(newly_tainted)
            if newly_tainted:
                print(f"Marked {len(newly_tainted)} clouds as tainted due to merging with incomplete tracks")

            # forward tainting — if any tainted track merged into a target cloud, taint the target cloud too
            def _collect_merge_recipients_of_tainted(tracks, tainted_ids):
                """Return set of track_ids that received a merge from any tainted track."""
                recipients = set()
                for trackid, clouds in tracks.items():
                    if trackid not in tainted_ids:
                        continue
                    for cloud in clouds:
                        target = getattr(cloud, 'merged_into', None)
                        if target is not None:
                            recipients.add(target)
                return recipients

            tainted_targets = _collect_merge_recipients_of_tainted(
                cloud_tracker.cloud_tracks, cloud_tracker.tainted_tracks
            )

            # Apply forward tainting (count only truly new)
            new_targets = tainted_targets - cloud_tracker.tainted_tracks
            if new_targets:
                cloud_tracker.tainted_tracks.update(new_targets)
                tainted_count += len(new_targets)
                print(f"Forward-tainted {len(new_targets)} merge recipients (absorbed tainted parents)")

        # Assign a stable, unique index for each track for NetCDF output.
        for track_id in cloud_tracker.cloud_tracks:
            if track_id not in cloud_tracker.track_id_to_index:
                cloud_tracker.track_id_to_index[track_id] = cloud_tracker.next_index
                cloud_tracker.next_index += 1

        # Write cloud track information to NetCDF
        write_cloud_tracks_to_netcdf(
            cloud_tracker.get_tracks(), 
            cloud_tracker.track_id_to_index,
            cloud_tracker.tainted_tracks,
            cloud_field.env_mass_flux_per_level, 
            output_netcdf_path, 
            timestep, 
            cloud_field.zt,
            config.get('env_ring_max_distance', 3)
        )

        gc.collect()
        # Calculate and display time taken for this timestep
        stop = time.time()
        minutes_taken = (stop - start) / 60
        print(f"Time taken: {minutes_taken:.1f} minutes")
    
    # Final Tainting Step
    # Clouds still active at the final timestep also have an incomplete lifecycle.
    final_partial_tracks = [tid for tid, track in cloud_tracker.cloud_tracks.items() 
                           if track[-1].is_active and track[-1].timestep == total_timesteps - 1]
    
    # Mark as tainted instead of deleting
    cloud_tracker.tainted_tracks.update(final_partial_tracks)
    tainted_count += len(final_partial_tracks)
    
    print("-"*50)
    print(f"Total partial lifetime clouds marked: {tainted_count}")
    print(f"Valid complete lifecycle clouds: {len(cloud_tracker.cloud_tracks) - tainted_count}")
    print("Cloud tracking complete.")

    # Overwrite the 'valid_track' variable with the final tainted status
    # so that the final NetCDF file correctly flags all partial lifecycle tracks.
    write_cloud_tracks_to_netcdf(
        cloud_tracker.get_tracks(),
        cloud_tracker.track_id_to_index,
        cloud_tracker.tainted_tracks,
        cloud_field.env_mass_flux_per_level, 
        output_netcdf_path, 
        total_timesteps - 1, 
        cloud_field.zt,
        config.get('env_ring_max_distance', 3)
    )


def finalize_partial_lifetime_tracks(cloud_tracker, total_timesteps):
    """
    LEGACY FUNCTION: Not currently used. Kept for reference only.
    This functionality is now handled directly in the process_clouds function.
    """
    partial_ids = []
    merged_ids = []
    
    for t_id, track in cloud_tracker.get_tracks().items():
        if not track:
            continue
        
        t_first = track[0].timestep
        t_last = track[-1].timestep
        last_cloud = track[-1]
        
        # Check for partial lifetime
        if t_first == 0 or (last_cloud.is_active and t_last == total_timesteps - 1):
            partial_ids.append(t_id)
        
        # Check for merged clouds
        if not last_cloud.is_active and last_cloud.merged_into is not None:
            merged_ids.append(t_id)

    # Update flags in the NetCDF
    with Dataset(output_netcdf_path, "r+") as ds:
        valid_var = ds.variables['valid_track']
        merged_var = ds.variables['merged_into']
        
        for i, t_id in enumerate(cloud_tracker.get_tracks().keys()):
            # Handle partial lifetime flags
            if t_id in partial_ids:
                valid_var[i] = 0
            
            # Note: merged_into is already handled in the write_cloud_tracks_to_netcdf function


def main(delete_existing_file):
    """Main function to set up and run the cloud tracking process."""
    # Start total time timer
    total_start_time = time.time()
    
    # Delete the existing output file if the option is provided
    if delete_existing_file and os.path.exists(output_netcdf_path):
        os.remove(output_netcdf_path)
        print(f"Deleted existing file: {output_netcdf_path}")

    # Initialize CloudTracker
    cloud_tracker = CloudTracker(config)

    # Process clouds - partial lifetime clouds will be marked as tainted (not removed)
    process_clouds(cloud_tracker)
    
    # Calculate and display total time
    total_end_time = time.time()
    total_minutes = (total_end_time - total_start_time) / 60
    print(f"\nTotal execution time: {total_minutes:.1f} minutes")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cloud tracking script.")
    parser.add_argument('--delete', action='store_true', help="Delete the existing output netcdf file before running the script.")
    args = parser.parse_args()

    main(args.delete)


## set proper output file path (placeholder for later)
# import os
# import datetime
# today = datetime.date.today()
# output_folder = f'output/{today}'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# output_netcdf_path = f'{output_folder}/{total_timesteps}timesteps_{config["l_condition"]}l_condition.nc'
# print(f"Output netcdf file path: {output_netcdf_path}")
