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
    'l': 'rico.l.nc',  # Liquid water
    'u': 'rico.u.nc',  # u wind
    'v': 'rico.v.nc',  # v wind
    'w': 'rico.w.nc',  # w wind (vertical velocity)
    'p': 'rico.p.nc',  # Pressure
    't': 'rico.t.nc',  # Temperature
    'q': 'rico.q.nc',  # Total water
}

# Processing Options
total_timesteps = 15  # Number of timesteps to process

# Cloud Definition and Tracking Configuration
config = {
    # Cloud identification
    'min_size': 10,              # Minimum number of points for a cloud to be considered a cloud
    'l_condition': 0.001,      # kg/kg. Minimum liquid water content for a point to be a cloud.
    'w_condition': 0.0,          # m/s. Minimum vertical velocity for a point to be part of a cloud.
    'w_switch': False,           # If True, apply the 'w_condition' threshold.
    
    # Simulation parameters
    'timestep_duration': 60,     # Seconds. Time between timesteps.
    'horizontal_resolution': 25.0, # m. Grid resolution from the simulation namelist.

    # Physics-based Adjustments
    'switch_wind_drift': True,   # If True, consider the mean horizontal motion in tracking.
    'switch_vertical_drift': True, # If True, consider the mean vertical motion in tracking.
    'cloud_base_altitude': 700,  # m. Estimated cloud base altitude for certain calculations (kind of deprecated)

    # Matching parameters
    'distance_threshold': 0,     # Max distance between merging clouds across a periodic boundary.
    'match_safety_factor': 2.0,  # Safety factor for matching clouds based on point overlap.
    'bounding_box_safety_factor': 2.0, # Safety factor for pre-filtering potential matches (using centroids).
    'max_expected_cloud_speed': 30.0,  # m/s. An estimate to constrain the search space for matching.
    'use_pre_filtering': True,   # If True, use a pre-filtering step to find potential matches (speed up matching).

    # Visualisation (somewhat deprecated)
    'plot_switch': False,        # If True, plot the cloud field at each timestep.
}
# --- End of user modifiable parameters ---



# Function to calculate mean_w
def calculate_mean_vertical_velocity(file_path, file_names, timestep):
    """Calculate mean vertical velocity for each z-level at the given timestep."""
    w_data = data_management.load_w_field(file_path, file_names['w'], timestep)
    mean_w = np.mean(w_data, axis=(1, 2))  # Average over y and x dimensions
    return mean_w

# @profile
def process_clouds(cloud_tracker):
    """Process clouds and mark partial lifecycle clouds as tainted."""
    tainted_count = 0
    
    for timestep in range(total_timesteps):
        start = time.time()
        print("-"*50)
        print(f"Processing timestep {timestep+1} of {total_timesteps}")

        # Load cloud field for the current timestep
        cloud_field = data_management.load_cloud_field_from_file(
            base_file_path, file_name, timestep, config
        )

        # Core tracking step: match current clouds to existing tracks.
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
            cloud_field.zt
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
        cloud_field.zt
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


