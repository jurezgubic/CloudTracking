from netCDF4 import Dataset
import numpy as np
import os

def initialize_netcdf(file_path, zt):
    """Create a new NetCDF file with the necessary dimensions and variables for cloud tracking data."""
    # Create the file and define dimensions and variables
    with Dataset(file_path, 'w', format='NETCDF4') as root_grp:
        root_grp.createDimension('track', 10000)  # Fixed large number of tracks
        root_grp.createDimension('time', None)  # Unlimited time dimension
        #root_grp.createDimension('point', 100000)  # Static dimension for cloud points Warning: Not in use, crude test remnant!
        root_grp.createDimension('coordinate', 3)  # Static dimension for 3D coordinates
        root_grp.createDimension('level', len(zt))  # Using consistent height levels

        # --- Add the track_id variable to store string identifiers ---
        track_id_str_var = root_grp.createVariable('track_id', str, ('track',))
        track_id_str_var.long_name = "String identifier for each cloud track"
        track_id_str_var.description = "Unique ID for each track, typically in 'timestep-label' format from its first appearance."

        # Flag for whether a track is fully valid or not
        valid_track_var = root_grp.createVariable('valid_track', 'i4', ('track',))
        valid_track_var[:] = 1  # Default all to valid

        root_grp.createVariable('size', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('max_height', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('cloud_base_area', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('max_w', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('max_w_cloud_base', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('surface_area', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('ql_flux', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('mass_flux', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('mass_flux_per_level', 'f4', ('track', 'time', 'level'), fill_value=np.nan)
        root_grp.createVariable('temp_per_level', 'f4', ('track', 'time', 'level'), fill_value=np.nan)
        root_grp.createVariable('theta_outside_per_level', 'f4', ('track', 'time', 'level'), fill_value=np.nan)
        root_grp.createVariable('w_per_level', 'f4', ('track', 'time', 'level'), fill_value=np.nan)
        root_grp.createVariable('circum_per_level', 'f4', ('track', 'time', 'level'), fill_value=np.nan)
        root_grp.createVariable('eff_radius_per_level', 'f4', ('track', 'time', 'level'), fill_value=np.nan)
        root_grp.createVariable('location_x', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('location_y', 'f4', ('track', 'time'), fill_value=np.nan)
        root_grp.createVariable('location_z', 'f4', ('track', 'time'), fill_value=np.nan)
        #root_grp.createVariable('cloud_points', 'f4', ('track', 'time', 'point', 'coordinate'), fill_value=np.nan)
        root_grp.createVariable('age', 'i4', ('track', 'time'), fill_value=-1)  # Add age variable

        # Add merged_into variable to track which clouds merged into others
        merged_into_var = root_grp.createVariable('merged_into', 'i4', ('track', 'time'), fill_value=-1)
        merged_into_var.long_name = "NetCDF index of the track this cloud merged into"
        merged_into_var.comment = "-1 if not merged, -2 if merged but target track index not found."

        height_var = root_grp.createVariable('height', 'f4', ('level', ))
        height_var[:] = zt  # Assign the height values


def write_cloud_tracks_to_netcdf(tracks, file_path, max_processed_timestep_idx, zt): # Renamed 'timestep' for clarity
    """Write cloud tracking data to a NetCDF file.
    This version writes the full history of each track.
    """

    if not os.path.exists(file_path):
        initialize_netcdf(file_path, zt)

    with Dataset(file_path, 'a') as root_grp:
        # --- Get the track_id string variable ---
        track_id_str_var = root_grp.variables['track_id']
        
        size_var = root_grp.variables['size']
        max_height_var = root_grp.variables['max_height']
        surface_area_var = root_grp.variables['surface_area']
        cloud_base_area_var = root_grp.variables['cloud_base_area']
        max_w_var = root_grp.variables['max_w']
        max_w_cloud_base_var = root_grp.variables['max_w_cloud_base']
        ql_flux_var = root_grp.variables['ql_flux']
        mass_flux_var = root_grp.variables['mass_flux']
        mass_flux_per_level_var = root_grp.variables['mass_flux_per_level']
        temp_per_level_var = root_grp.variables['temp_per_level']
        theta_outside_per_level_var = root_grp.variables['theta_outside_per_level']
        w_per_level_var = root_grp.variables['w_per_level']
        circum_per_level_var = root_grp.variables['circum_per_level']
        eff_radius_per_level_var = root_grp.variables['eff_radius_per_level']
        loc_x_var = root_grp.variables['location_x']
        loc_y_var = root_grp.variables['location_y']
        loc_z_var = root_grp.variables['location_z']
        #cloud_points_var = root_grp.variables['cloud_points']
        age_var = root_grp.variables['age']
        valid_track_var = root_grp.variables['valid_track']  # We can reference it if needed
        merged_into_var = root_grp.variables['merged_into']

        active_track_ids_list = list(tracks.keys()) # List of string track IDs, defines order for NetCDF track dimension

        for i, track_id_str in enumerate(active_track_ids_list): # 'i' is the NetCDF track dimension index
            # --- Populate the track_id string variable ---
            track_id_str_var[i] = track_id_str

            # Iterate through each cloud object (timestep state) in the track's history
            for cloud_obj in tracks[track_id_str]:
                t_idx = cloud_obj.timestep # The timestep index for this specific cloud state

                # Write data for this cloud state at its specific timestep
                size_var[i, t_idx] = cloud_obj.size
                max_height_var[i, t_idx] = cloud_obj.max_height
                max_w_var[i, t_idx] = cloud_obj.max_w
                max_w_cloud_base_var[i, t_idx] = cloud_obj.max_w_cloud_base
                ql_flux_var[i, t_idx] = cloud_obj.ql_flux
                mass_flux_var[i, t_idx] = cloud_obj.mass_flux
                if cloud_obj.mass_flux_per_level is not None and len(cloud_obj.mass_flux_per_level) == len(zt):
                    mass_flux_per_level_var[i, t_idx, :] = cloud_obj.mass_flux_per_level
                if cloud_obj.temp_per_level is not None and len(cloud_obj.temp_per_level) == len(zt):
                    temp_per_level_var[i, t_idx, :] = cloud_obj.temp_per_level
                if cloud_obj.theta_outside_per_level is not None and len(cloud_obj.theta_outside_per_level) == len(zt):
                    theta_outside_per_level_var[i, t_idx, :] = cloud_obj.theta_outside_per_level
                if cloud_obj.w_per_level is not None and len(cloud_obj.w_per_level) == len(zt):
                    w_per_level_var[i, t_idx, :] = cloud_obj.w_per_level
                if cloud_obj.circum_per_level is not None and len(cloud_obj.circum_per_level) == len(zt):
                    circum_per_level_var[i, t_idx, :] = cloud_obj.circum_per_level
                if cloud_obj.eff_radius_per_level is not None and len(cloud_obj.eff_radius_per_level) == len(zt):
                    eff_radius_per_level_var[i, t_idx, :] = cloud_obj.eff_radius_per_level
                cloud_base_area_var[i, t_idx] = cloud_obj.cloud_base_area
                surface_area_var[i, t_idx] = cloud_obj.surface_area
                loc_x_var[i, t_idx], loc_y_var[i, t_idx], loc_z_var[i, t_idx] = cloud_obj.location
                age_var[i, t_idx] = cloud_obj.age

                # Handle merged_into for this cloud state
                if cloud_obj.merged_into:
                    target_track_id_str = cloud_obj.merged_into
                    if target_track_id_str in active_track_ids_list:
                        merged_target_netcdf_idx = active_track_ids_list.index(target_track_id_str)
                        merged_into_var[i, t_idx] = merged_target_netcdf_idx
                    else:
                        merged_into_var[i, t_idx] = -2 # Merged, but target track not in current list
                else:
                    merged_into_var[i, t_idx] = -1 # Not merged

            # If the track is no longer active, ensure subsequent timesteps are NaN
            # This is implicitly handled if fill_value is np.nan and we only write actual data points.
            # However, if a track becomes inactive and then somehow data was written for it later (should not happen),
            # this explicit step could be useful. For now, relying on fill_value.
            last_cloud_obj_in_track = tracks[track_id_str][-1]
            if not last_cloud_obj_in_track.is_active:
                start_nan_t_idx = last_cloud_obj_in_track.timestep + 1
                if start_nan_t_idx <= max_processed_timestep_idx:
                    # Explicitly fill NaNs for remaining timesteps for this inactive track
                    # This ensures that if the NetCDF file was somehow pre-filled or reused,
                    # old data for these time slots is cleared.
                    # ToDo (optimisation): Slicing with a variable upper bound for time might be slow
                    # if max_processed_timestep_idx is very large and 'time' is truly unlimited.
                    # For now, assume max_processed_timestep_idx is reasonable.
                    for t_fill_idx in range(start_nan_t_idx, max_processed_timestep_idx + 1):
                        size_var[i, t_fill_idx] = np.nan
                        max_height_var[i, t_fill_idx] = np.nan
                        max_w_var[i, t_fill_idx] = np.nan
                        max_w_cloud_base_var[i, t_fill_idx] = np.nan
                        ql_flux_var[i, t_fill_idx] = np.nan
                        mass_flux_var[i, t_fill_idx] = np.nan
                        # For 3D vars, assign slice of NaNs
                        nan_level_array = np.full(len(zt), np.nan)
                        mass_flux_per_level_var[i, t_fill_idx, :] = nan_level_array
                        temp_per_level_var[i, t_fill_idx, :] = nan_level_array
                        theta_outside_per_level_var[i, t_fill_idx, :] = nan_level_array
                        w_per_level_var[i, t_fill_idx, :] = nan_level_array
                        circum_per_level_var[i, t_fill_idx, :] = nan_level_array
                        eff_radius_per_level_var[i, t_fill_idx, :] = nan_level_array
                        cloud_base_area_var[i, t_fill_idx] = np.nan
                        surface_area_var[i, t_fill_idx] = np.nan
                        loc_x_var[i, t_fill_idx] = np.nan
                        loc_y_var[i, t_fill_idx] = np.nan
                        loc_z_var[i, t_fill_idx] = np.nan
                        age_var[i, t_fill_idx] = -1
                        merged_into_var[i, t_fill_idx] = -1 # Not merged in these future NaNs

