from netCDF4 import Dataset
import numpy as np
import os

def initialize_netcdf(file_path, zt, ring_count, env_aloft_levels=40):
    """Create a new NetCDF file with necessary dimensions and variables for cloud tracking data."""
    # Create the file and define dimensions and variables
    with Dataset(file_path, 'w', format='NETCDF4') as root_grp:
        root_grp.createDimension('track', 100000)  # Fixed large number of tracks
        root_grp.createDimension('time', None)  # Unlimited time dimension
        #root_grp.createDimension('point', 100000)  # Static dimension for cloud points Warning: Not in use, crude test remnant!
        root_grp.createDimension('coordinate', 3)  # Static dimension for 3D coordinates
        root_grp.createDimension('level', len(zt))  # Using consistent height levels
        root_grp.createDimension('ring', ring_count)  # Environment ring distance index (1..D)
        
        # Environment Aloft Dimension
        root_grp.createDimension('height_aloft', env_aloft_levels)

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
        # NIP diagnostics
        root_grp.createVariable('nip_per_level', 'f4', ('track', 'time', 'level'), fill_value=np.nan)
        root_grp.createVariable('nip_acc_per_level', 'f4', ('track', 'time', 'level'), fill_value=np.nan)
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

        root_grp.createVariable('area_per_level', 'f4', ('track','time','level'), fill_value=np.nan)
        root_grp.createVariable('equiv_radius_per_level', 'f4', ('track','time','level'), fill_value=np.nan)
        root_grp.createVariable('compactness_per_level', 'f4', ('track','time','level'), fill_value=np.nan)
        root_grp.createVariable('base_radius_prescribed', 'f4', ('track','time'), fill_value=np.nan)
        root_grp.createVariable('base_radius_diagnosed', 'f4', ('track','time'), fill_value=np.nan)
        root_grp.createVariable('base_area_diagnosed', 'f4', ('track','time'), fill_value=np.nan)
        root_grp.createVariable('max_equiv_radius', 'f4', ('track','time'), fill_value=np.nan)

        # Environment ring variables (track, time, level, ring)
        root_grp.createVariable('env_w_rings', 'f4', ('track','time','level','ring'), fill_value=np.nan)
        root_grp.createVariable('env_l_rings', 'f4', ('track','time','level','ring'), fill_value=np.nan)
        root_grp.createVariable('env_qt_rings', 'f4', ('track','time','level','ring'), fill_value=np.nan)
        root_grp.createVariable('env_qv_rings', 'f4', ('track','time','level','ring'), fill_value=np.nan)
        root_grp.createVariable('env_p_rings', 'f4', ('track','time','level','ring'), fill_value=np.nan)
        root_grp.createVariable('env_theta_l_rings', 'f4', ('track','time','level','ring'), fill_value=np.nan)
        root_grp.createVariable('env_buoyancy_rings', 'f4', ('track','time','level','ring'), fill_value=np.nan)

        # Environment Aloft variables (track, time, height_aloft)
        root_grp.createVariable('env_aloft_qt_diff', 'f4', ('track', 'time', 'height_aloft'), fill_value=np.nan)
        root_grp.createVariable('env_aloft_thetal_diff', 'f4', ('track', 'time', 'height_aloft'), fill_value=np.nan)
        root_grp.createVariable('env_aloft_shear', 'f4', ('track', 'time', 'height_aloft'), fill_value=np.nan)
        root_grp.createVariable('env_aloft_n2', 'f4', ('track', 'time', 'height_aloft'), fill_value=np.nan)
        root_grp.createVariable('env_aloft_rh', 'f4', ('track', 'time', 'height_aloft'), fill_value=np.nan)


def write_cloud_tracks_to_netcdf(tracks, track_id_to_index, tainted_tracks, env_mass_flux_per_level, file_path, timestep, zt, ring_count, env_aloft_levels=40):
    """Write cloud tracking data (including environment rings) to NetCDF for a timestep."""

    # Create the file if it doesn't exist
    if not os.path.exists(file_path):
        initialize_netcdf(file_path, zt, ring_count, env_aloft_levels)

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
        area_per_level_var = root_grp.variables['area_per_level']
        equiv_radius_per_level_var = root_grp.variables['equiv_radius_per_level']
        compactness_per_level_var = root_grp.variables['compactness_per_level']
        base_radius_prescribed_var = root_grp.variables['base_radius_prescribed']
        base_radius_diagnosed_var = root_grp.variables['base_radius_diagnosed']
        base_area_diagnosed_var = root_grp.variables['base_area_diagnosed']
        max_equiv_radius_var = root_grp.variables['max_equiv_radius']

        # Environment ring variables
        env_w_rings_var = root_grp.variables['env_w_rings']
        env_l_rings_var = root_grp.variables['env_l_rings']
        env_qt_rings_var = root_grp.variables['env_qt_rings']
        env_qv_rings_var = root_grp.variables['env_qv_rings']
        env_p_rings_var = root_grp.variables['env_p_rings']
        env_theta_l_rings_var = root_grp.variables['env_theta_l_rings']
        env_buoyancy_rings_var = root_grp.variables['env_buoyancy_rings']

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
                cloud = clouds_at_timestep[0]
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
                # NIP diagnostics (if computed)
                root_grp.variables['nip_per_level'][i, timestep, :] = cloud.nip_per_level
                root_grp.variables['nip_acc_per_level'][i, timestep, :] = cloud.nip_acc_per_level
                cloud_base_area_var[i, timestep] = cloud.cloud_base_area
                surface_area_var[i, timestep] = cloud.surface_area
                loc_x_var[i, timestep], loc_y_var[i, timestep], loc_z_var[i, timestep] = cloud.location
                #cloud_points_var[i, timestep, :len(points), :] = points
                age_var[i, timestep] = cloud.age
                cloud_base_height_var[i, timestep] = cloud.cloud_base_height
                area_per_level_var[i, timestep, :] = cloud.area_per_level
                equiv_radius_per_level_var[i, timestep, :] = cloud.equiv_radius_per_level
                compactness_per_level_var[i, timestep, :] = cloud.compactness_per_level
                base_radius_prescribed_var[i, timestep] = (np.sqrt(cloud.cloud_base_area/np.pi)
                                                           if cloud.cloud_base_area > 0 else np.nan)
                base_radius_diagnosed_var[i, timestep] = cloud.base_radius_diagnosed
                base_area_diagnosed_var[i, timestep] = cloud.base_area_diagnosed
                max_equiv_radius_var[i, timestep] = cloud.max_equiv_radius

                # Write merge and split tracking data
                merges_count_var[i, timestep] = cloud.merges_count
                splits_count_var[i, timestep] = cloud.splits_count

                # Write environment ring arrays if present
                if getattr(cloud, 'env_w_rings', None) is not None:
                    env_w_rings_var[i, timestep, :, :] = cloud.env_w_rings
                    env_l_rings_var[i, timestep, :, :] = cloud.env_l_rings
                    env_qt_rings_var[i, timestep, :, :] = cloud.env_qt_rings
                    env_qv_rings_var[i, timestep, :, :] = cloud.env_qv_rings
                    env_p_rings_var[i, timestep, :, :] = cloud.env_p_rings
                    env_theta_l_rings_var[i, timestep, :, :] = cloud.env_theta_l_rings
                    env_buoyancy_rings_var[i, timestep, :, :] = cloud.env_buoyancy_rings
                
                # Write environment aloft arrays if present
                if getattr(cloud, 'env_aloft_qt_diff', None) is not None:
                    root_grp.variables['env_aloft_qt_diff'][i, timestep, :] = cloud.env_aloft_qt_diff
                    root_grp.variables['env_aloft_thetal_diff'][i, timestep, :] = cloud.env_aloft_thetal_diff
                    root_grp.variables['env_aloft_shear'][i, timestep, :] = cloud.env_aloft_shear
                    root_grp.variables['env_aloft_n2'][i, timestep, :] = cloud.env_aloft_n2
                    root_grp.variables['env_aloft_rh'][i, timestep, :] = cloud.env_aloft_rh

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
                # if a merge occurred at t-1, record it now
                if timestep > 0 and track:
                    last_cloud = track[-1]
                    if (last_cloud.timestep == timestep - 1) and (not last_cloud.is_active) and (last_cloud.merged_into is not None):
                        if last_cloud.merged_into in track_id_to_index:
                            merged_into_var[i, timestep - 1] = track_id_to_index[last_cloud.merged_into]
                        else:
                            merged_into_var[i, timestep - 1] = -2  # Unknown target

                # Optimization: Do not write NaNs for inactive tracks.
                # The file is initialized with fill_value=NaN (or -1), so we don't need to explicitly write them.
                pass
