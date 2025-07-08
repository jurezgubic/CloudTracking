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
        root_grp.createVariable('env_mass_flux_per_level', 'f4', ('time', 'level'), fill_value=np.nan)
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

        height_var = root_grp.createVariable('height', 'f4', ('level', ))
        height_var[:] = zt  # Assign the height values


def write_cloud_tracks_to_netcdf(tracks, env_mass_flux_per_level, file_path, timestep, zt):
    """Write cloud tracking data to a NetCDF file for a given timestep."""

    # Create the file. Warning: This will overwrite the file if it already exists.
    if not os.path.exists(file_path):
        initialize_netcdf(file_path, zt)

    # Write data for active clouds at this timestep
    with Dataset(file_path, 'a') as root_grp:
        size_var = root_grp.variables['size']
        max_height_var = root_grp.variables['max_height']
        surface_area_var = root_grp.variables['surface_area']
        cloud_base_area_var = root_grp.variables['cloud_base_area']
        max_w_var = root_grp.variables['max_w']
        max_w_cloud_base_var = root_grp.variables['max_w_cloud_base']
        ql_flux_var = root_grp.variables['ql_flux']
        mass_flux_var = root_grp.variables['mass_flux']
        mass_flux_per_level_var = root_grp.variables['mass_flux_per_level']
        env_mass_flux_per_level_var = root_grp.variables['env_mass_flux_per_level']
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

        # Write environment data for this timestep
        if env_mass_flux_per_level is not None:
            env_mass_flux_per_level_var[timestep, :] = env_mass_flux_per_level

        # Write data for active clouds at this timestep
        active_tracks = list(tracks.keys())
        for i, track_id in enumerate(active_tracks):
            cloud = tracks[track_id][-1]  # Get the last cloud which represents the current state
            if cloud.is_active:
                size_var[i, timestep] = cloud.size
                max_height_var[i, timestep] = cloud.max_height
                max_w_var[i, timestep] = cloud.max_w
                max_w_cloud_base_var[i, timestep] = cloud.max_w_cloud_base
                ql_flux_var[i, timestep] = cloud.ql_flux
                mass_flux_var[i, timestep] = cloud.mass_flux
                mass_flux_per_level_var[i, timestep, :] = cloud.mass_flux_per_level
                temp_per_level_var[i, timestep, :] = cloud.temp_per_level
                theta_outside_per_level_var[i, timestep, :] = cloud.theta_outside_per_level
                w_per_level_var[i, timestep, :] = cloud.w_per_level
                circum_per_level_var[i, timestep, :] = cloud.circum_per_level
                eff_radius_per_level_var[i, timestep, :] = cloud.eff_radius_per_level
                cloud_base_area_var[i, timestep] = cloud.cloud_base_area
                surface_area_var[i, timestep] = cloud.surface_area
                loc_x_var[i, timestep], loc_y_var[i, timestep], loc_z_var[i, timestep] = cloud.location
                #points = np.array([list(p) for p in cloud.points[:10000]])
                #cloud_points_var[i, timestep, :len(points), :] = points
                age_var[i, timestep] = cloud.age
            else:
                # Set current and future entries to NaN for inactive clouds
                size_var[i, timestep:] = np.nan
                max_height_var[i, timestep:] = np.nan
                max_w_var[i, timestep:] = np.nan
                max_w_cloud_base_var[i, timestep:] = np.nan
                ql_flux_var[i, timestep:] = np.nan
                mass_flux_var[i, timestep:] = np.nan
                mass_flux_per_level_var[i, timestep:, :] = np.nan
                temp_per_level_var[i, timestep:, :] = np.nan
                theta_outside_per_level_var[i, timestep:, :] = np.nan
                w_per_level_var[i, timestep:, :] = np.nan
                circum_per_level_var[i, timestep:, :] = np.nan
                eff_radius_per_level_var[i, timestep:, :] = np.nan
                cloud_base_area_var[i, timestep:] = np.nan
                surface_area_var[i, timestep:] = np.nan
                loc_x_var[i, timestep:] = np.nan
                loc_y_var[i, timestep:] = np.nan
                loc_z_var[i, timestep:] = np.nan
                #cloud_points_var[i, timestep:, :, :] = np.nan
                age_var[i, timestep:] = -1

                # Write merged_into information
                if not cloud.is_active and cloud.merged_into is not None:
                    # Find the index of the track it merged into
                    if cloud.merged_into in active_tracks:
                        merged_idx = active_tracks.index(cloud.merged_into)
                        merged_into_var[i, timestep] = merged_idx
                    else:
                        # If the track it merged into isn't in our list (shouldn't happen)
                        merged_into_var[i, timestep] = -2  # Special value indicating merged but target unknown
                else:
                    merged_into_var[i, timestep] = -1  # -1 means not merged

