from netCDF4 import Dataset
import numpy as np
import os

def initialize_netcdf(file_path, zt):
    """Create a new NetCDF file with necessary dimensions and variables for cloud tracking data."""
    # Create the file and define dimensions and variables
    with Dataset(file_path, 'w', format='NETCDF4') as root_grp:
        root_grp.createDimension('track', 100000)  # Fixed large number of tracks
        root_grp.createDimension('time', None)  # Unlimited time dimension
        #root_grp.createDimension('point', 100000)  # Static dimension for cloud points Warning: Not in use, crude test remnant!
        root_grp.createDimension('coordinate', 3)  # Static dimension for 3D coordinates
        root_grp.createDimension('level', len(zt))  # Using consistent height levels

        # Flag for whether a track is fully valid or not (0=partial, 1=complete)
        valid_track_var = root_grp.createVariable('valid_track', 'i4', ('track',))
        valid_track_var[:] = 1  # Default all to valid

        # Track ID variable to help with debugging and analysis
        track_id_var = root_grp.createVariable('track_id', 'i8', ('track',))
        
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

        merges_count_var = root_grp.createVariable('merges_count', 'i4', ('track', 'time'), fill_value=0)
        splits_count_var = root_grp.createVariable('splits_count', 'i4', ('track', 'time'), fill_value=0)
        split_from_var = root_grp.createVariable('split_from', 'i4', ('track', 'time'), fill_value=-1)

        height_var = root_grp.createVariable('height', 'f4', ('level', ))
        height_var[:] = zt  # Assign the height values

        # Add cloud base height variable
        root_grp.createVariable('cloud_base_height', 'f4', ('track', 'time'), fill_value=np.nan)


def write_cloud_tracks_to_netcdf(tracks, track_id_to_index, tainted_tracks, env_mass_flux_per_level, file_path, timestep, zt):
    """Write cloud tracking data to a NetCDF file for a given timestep using stable indices."""

    # Create the file if it doesn't exist
    if not os.path.exists(file_path):
        initialize_netcdf(file_path, zt)

    # Write data for clouds at this timestep
    with Dataset(file_path, 'a') as root_grp:
        # Get variable handles
        size_var = root_grp.variables['size']
        max_height_var = root_grp.variables['max_height']
        surface_area_var = root_grp.variables['surface_area']
        cloud_base_area_var = root_grp.variables['cloud_base_area']
        cloud_base_height_var = root_grp.variables['cloud_base_height']
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
        track_id_var = root_grp.variables['track_id']  # New variable for track IDs

        merges_count_var = root_grp.variables['merges_count']
        splits_count_var = root_grp.variables['splits_count']
        split_from_var = root_grp.variables['split_from']

        # Write environment data for this timestep
        if env_mass_flux_per_level is not None:
            env_mass_flux_per_level_var[timestep, :] = env_mass_flux_per_level

        # Write track IDs only once (timestep 0 or whenever a track is first seen)
        for track_id, idx in track_id_to_index.items():
            if idx < len(track_id_var):  # Ensure we don't exceed array bounds
                # Try to convert string IDs to integers for storage
                try:
                    numeric_id = int(track_id.split('-')[-1]) 
                    track_id_var[idx] = numeric_id
                except:
                    # If conversion fails, just store -1 or another sentinel
                    track_id_var[idx] = -1

        # Write data for all tracks
        for track_id, track in tracks.items():
            if not track:  # Skip empty tracks
                continue
                
            # Use the stable index
            i = track_id_to_index[track_id]
            
            # Mark tainted tracks (partial lifecycle)
            if track_id in tainted_tracks:
                valid_track_var[i] = 0
            
            # Get the cloud state at this timestep
            clouds_at_timestep = [c for c in track if c.timestep == timestep]
            
            if clouds_at_timestep:
                cloud = clouds_at_timestep[0]  # There should be at most one cloud per track per timestep
                
                # Write cloud data to its stable index location
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
                #cloud_points_var[i, timestep, :len(points), :] = points
                age_var[i, timestep] = cloud.age
                cloud_base_height_var[i, timestep] = cloud.cloud_base_height
                
                # Write merge and split tracking data
                merges_count_var[i, timestep] = cloud.merges_count
                splits_count_var[i, timestep] = cloud.splits_count
                
                # Write split_from information if available
                if hasattr(cloud, 'split_from') and cloud.split_from is not None:
                    # Convert track ID to index
                    if cloud.split_from in track_id_to_index:
                        split_from_var[i, timestep] = track_id_to_index[cloud.split_from]
                    else:
                        split_from_var[i, timestep] = -2  # Unknown parent track

                # Move this check INSIDE the if-block to prevent UnboundLocalError
                if not cloud.is_active and cloud.merged_into is not None:
                    if cloud.merged_into in track_id_to_index:
                        merged_idx = track_id_to_index[cloud.merged_into]
                        merged_into_var[i, timestep] = merged_idx
                    else:
                        # If the target track doesn't have an index yet (shouldn't happen)
                        merged_into_var[i, timestep] = -2
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

                # Instead, simply set the merged_into field to the default "not merged" value
                merged_into_var[i, timestep] = -1  # -1 means not merged

