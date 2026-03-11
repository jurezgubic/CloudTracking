import argparse
import gc
import os
import time

import numpy as np
from netCDF4 import Dataset

import src.data_management as data_management
from lib.cloudtracker import CloudTracker
from src.config_loader import load_config
from src.data_management import (
    create_data_adapter,
    load_cloud_field_from_adapter,
)
from src.netcdf_writer import write_cloud_tracks_to_netcdf


# Function to calculate mean_w
def calculate_mean_vertical_velocity(file_path, file_names, timestep):
    """Calculate mean vertical velocity for each z-level at the given timestep."""
    w_data = data_management.load_w_field(file_path, file_names["w"], timestep)
    mean_w = np.mean(w_data, axis=(1, 2))  # Average over y and x dimensions
    return mean_w


def resolve_cloud_base_altitude(config, z_levels):
    """
    Resolve user-provided cloud_base_altitude (meters) to nearest model level.
    Stores:
      config['cloud_base_level_index']
      config['cloud_base_altitude_resolved']
    """
    target = config.get("cloud_base_altitude", None)
    if target is None:
        raise ValueError("config must contain 'cloud_base_altitude'")
    z_arr = np.asarray(z_levels, dtype=float)
    if z_arr.ndim != 1 or z_arr.size == 0:
        raise ValueError("z_levels must be a 1D non-empty array")
    idx = int(np.abs(z_arr - target).argmin())
    resolved = float(z_arr[idx])
    config["cloud_base_level_index"] = idx
    config["cloud_base_altitude_resolved"] = resolved
    print(f"Cloud base requested at {target}m. Nearest resolved (and used) height is {resolved}m.")


# @profile
def process_clouds(cloud_tracker, adapter, num_timesteps, config, output_netcdf_path):
    """Process clouds and mark partial lifecycle clouds as tainted.

    Parameters
    ----------
    cloud_tracker : CloudTracker
        The cloud tracker instance
    adapter : BaseDataAdapter
        Data adapter for loading cloud fields
    num_timesteps : int
        Number of timesteps to process
    """
    tainted_count = 0
    base_resolved = False

    for timestep in range(num_timesteps):
        start = time.time()
        print("-" * 50)
        print(f"Processing timestep {timestep + 1} of {num_timesteps}")

        # Load cloud field using adapter
        cloud_field = load_cloud_field_from_adapter(adapter, timestep, config)

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

                # Optimization: Only check the last cloud, as merged_into is set when a track ends
                if track:
                    last_cloud = track[-1]
                    if hasattr(last_cloud, "merged_into") and last_cloud.merged_into in cloud_tracker.tainted_tracks:
                        newly_tainted.append(track_id)

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
                    # Optimization: Only check the last cloud
                    if clouds:
                        last_cloud = clouds[-1]
                        target = getattr(last_cloud, "merged_into", None)
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
            config.get("env_ring_max_distance", 3),
            env_aloft_levels=config.get("env_aloft_levels", 40),
            config=config,
        )

        gc.collect()
        # Calculate and display time taken for this timestep
        stop = time.time()
        minutes_taken = (stop - start) / 60
        print(f"Time taken: {minutes_taken:.1f} minutes")

    # Final Tainting Step
    # Clouds still active at the final timestep also have an incomplete lifecycle.
    final_partial_tracks = [
        tid
        for tid, track in cloud_tracker.cloud_tracks.items()
        if track[-1].is_active and track[-1].timestep == num_timesteps - 1
    ]

    # Mark as tainted instead of deleting
    cloud_tracker.tainted_tracks.update(final_partial_tracks)

    print("-" * 50)
    tainted_count = len(cloud_tracker.tainted_tracks)
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
        num_timesteps - 1,
        cloud_field.zt,
        config.get("env_ring_max_distance", 3),
        env_aloft_levels=config.get("env_aloft_levels", 40),
        config=config,
    )


def finalize_partial_lifetime_tracks(cloud_tracker, total_timesteps, output_netcdf_path):
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
        valid_var = ds.variables["valid_track"]
        ds.variables["merged_into"]

        for i, t_id in enumerate(cloud_tracker.get_tracks().keys()):
            # Handle partial lifetime flags
            if t_id in partial_ids:
                valid_var[i] = 0

            # Note: merged_into is already handled in the write_cloud_tracks_to_netcdf function


def main(config, delete_existing_file):
    """Main function to set up and run the cloud tracking process."""
    # Start total time timer
    total_start_time = time.time()

    output_netcdf_path = config.get("output_path", "cloud_results.nc")
    total_timesteps = config.get("total_timesteps", -1)

    # Delete the existing output file if the option is provided
    if delete_existing_file and os.path.exists(output_netcdf_path):
        os.remove(output_netcdf_path)
        print(f"Deleted existing file: {output_netcdf_path}")

    # Create data adapter based on configuration
    print(f"\nInitializing {config['data_format']} data adapter...")
    adapter = create_data_adapter(config)

    # Determine number of timesteps to process
    available_timesteps = adapter.get_total_timesteps()
    if total_timesteps == -1 or total_timesteps > available_timesteps:
        num_timesteps = available_timesteps
        print(f"Processing all {num_timesteps} available timesteps")
    else:
        num_timesteps = total_timesteps
        print(f"Processing {num_timesteps} of {available_timesteps} available timesteps")

    # Get grid info and update config with horizontal resolution
    grid_info = adapter.get_grid_info()
    config["horizontal_resolution"] = grid_info["dx"]
    print(f"Horizontal resolution: {config['horizontal_resolution']} m")

    # Initialize CloudTracker
    cloud_tracker = CloudTracker(config)

    # Process clouds - partial lifetime clouds will be marked as tainted (not removed)
    process_clouds(cloud_tracker, adapter, num_timesteps, config, output_netcdf_path)

    # Calculate and display total time
    total_end_time = time.time()
    total_minutes = (total_end_time - total_start_time) / 60
    print(f"\nTotal execution time: {total_minutes:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cloud tracking script.")
    parser.add_argument(
        "--config",
        default="configs/rico.toml",
        help="Path to TOML config file (default: configs/rico.toml)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the existing output netcdf file before running the script.",
    )
    args = parser.parse_args()

    print(f"Loading config: {args.config}")
    config = load_config(args.config)
    main(config, args.delete)
