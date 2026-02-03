import numpy as np
import gc
from skimage import measure
from memory_profiler import profile
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from utils.plotting_utils import plot_labeled_regions
from lib.cloud import Cloud
from utils.physics import calculate_physics_variables, calculate_rh_and_temperature
from scipy.spatial import cKDTree
import utils.constants as const

class CloudField:
    """Class to identify and track clouds in a labeled data field."""
    def __init__(self, l_data, u_data, v_data, w_data, p_data, theta_l_data, q_t_data, r_data, timestep, config, xt, yt, zt):
        self.timestep = timestep
        self.distance_threshold = config['distance_threshold']
        self.xt = xt
        self.yt = yt
        self.zt = zt
        self.env_mass_flux_per_level = None

        # creates a labeled array of objects
        labeled_array = self.identify_regions(l_data, w_data, config)

        if config['plot_switch'] == True:
            plot_labeled_regions(
                'initial_labeled_array', labeled_array, timestep=timestep, plot_all_levels=False, specific_level=50)

        # find merges between edge regions
        merges = self.find_boundary_merges(labeled_array)

        # update labels for merges
        updated_labeled_array = self.update_labels_for_merges(labeled_array, merges)

        # Calculate environment mass flux before creating cloud objects
        self.env_mass_flux_per_level = self._calculate_environment_mass_flux(
            updated_labeled_array, p_data, theta_l_data, l_data, q_t_data, r_data, w_data, config
        )

        # Calculate domain mean profiles for environment aloft analysis
        self.mean_theta_l_profile = np.mean(theta_l_data, axis=(1, 2))
        self.mean_qt_profile = np.mean(q_t_data, axis=(1, 2))

        # create cloud data from updated labeled array
        self.clouds = self.create_clouds_from_labeled_array(
            updated_labeled_array, l_data, u_data, v_data, w_data, p_data, theta_l_data, q_t_data, r_data, config, xt, yt, zt)

        # plot the updated labeled clouds if plot_switch is True
        if config['plot_switch'] == True:
            plot_labeled_regions(
                'updated_labeled_array', updated_labeled_array, timestep=timestep, plot_all_levels=False, specific_level=50)

        # example of how to plot labeled regions
        # plot_labeled_regions(
            #'name_or_array', labeled_array, timestep=timestep, plot_all_levels=False, specific_level=50)
        # plot_labeled_regions(
            #'name_of_array', labeled_array, timestep=timestep, plot_all_levels=True) #plots all levels

        # Add these attributes to store the precomputed KD-tree
        self.surface_points_array = None
        self.surface_point_to_cloud_id = None
        self.surface_points_kdtree = None

        # After clouds are created
        self.build_global_surface_kdtree()

        # Compute Neighbour Interaction Potential (NIP) for all clouds in this field
        self._compute_nip(config)

    # ---------------------------
    # Instantaneous environment mass flux
    # ---------------------------

    def _calculate_environment_mass_flux(self, labeled_array, p_data, theta_l_data, l_data, q_t_data, r_data, w_data, config):
        """Calculate mass flux and baseline densities for the environment at each level."""
        print("Calculating environment mass flux...")
        env_mask = labeled_array == 0
        n_levels = len(self.zt)
        
        if not np.any(env_mask):
            # No environment present: set empty fields
            self.env_rho_mean_per_level = np.full(n_levels, np.nan)
            return np.zeros(n_levels)

        # Get coordinates of all environment points
        z_indices, _, _ = np.where(env_mask)

        # Extract 1D arrays of physical variables for all environment points
        p_values_env = p_data[env_mask]
        theta_l_values_env = theta_l_data[env_mask]
        l_values_env = l_data[env_mask]
        q_t_values_env = q_t_data[env_mask]
        r_values_env = r_data[env_mask]
        w_values_env = w_data[env_mask]

        # Note: RICO data water species (l, q, r) are already in kg/kg (metadata labels g/kg incorrectly)
        q_l_values_env = l_values_env
        q_v_values_env = q_t_values_env - q_l_values_env

        # Convert any masked arrays to regular arrays (needed for Numba)
        if hasattr(p_values_env, 'filled'):
            p_values_env = p_values_env.filled(np.nan)
        if hasattr(theta_l_values_env, 'filled'):
            theta_l_values_env = theta_l_values_env.filled(np.nan)
        if hasattr(q_l_values_env, 'filled'):
            q_l_values_env = q_l_values_env.filled(np.nan)
        if hasattr(q_v_values_env, 'filled'):
            q_v_values_env = q_v_values_env.filled(np.nan)
        if hasattr(r_values_env, 'filled'):
            r_values_env = r_values_env.filled(np.nan)
        if hasattr(w_values_env, 'filled'):
            w_values_env = w_values_env.filled(np.nan)

        # Reuse the existing physics function to get densities and mass flux for each point
        # Rain is excluded from saturation adjustment but included in density loading
        _, rhos_env, mass_fluxes_env = calculate_physics_variables(
            p_values_env,
            theta_l_values_env,
            q_l_values_env,
            q_v_values_env,
            w_values_env,
            config['horizontal_resolution']**2,
            r_rain_values=r_values_env
        )

        # Sum the mass fluxes per level using the z-indices
        # np.bincount is highly efficient for this aggregation task
        env_mass_flux_per_level = np.bincount(z_indices, weights=mass_fluxes_env, minlength=n_levels)
        # Compute mean density per level for environment (used for buoyancy reference)
        counts = np.bincount(z_indices, minlength=n_levels).astype(float)
        sum_rho = np.bincount(z_indices, weights=rhos_env, minlength=n_levels)
        with np.errstate(invalid='ignore', divide='ignore'):
            self.env_rho_mean_per_level = sum_rho / counts
        
        return env_mass_flux_per_level

    def identify_regions(self, l_data, w_data, config):
        """Identify cloudy regions in the labeled data."""
        if config['w_switch'] == True:
            condition = (l_data > config['l_condition']) & (w_data >= config['w_condition'])
        else:
            condition = l_data > config['l_condition']

        # Label the regions
        labeled_array = measure.label(condition, connectivity=3)
        num_features = np.max(labeled_array)
        print(f"Objects labeled. Number of initial features: {num_features}")

        return labeled_array



    def find_boundary_merges(self, labeled_array):
        """Find merges between regions on opposite boundaries of the labeled array."""
        merges = []
        z, y, x = labeled_array.shape
        threshold = self.distance_threshold

        # Function to find boundary points for a given label
        def boundary_points(label, boundary_array):
            return np.argwhere(boundary_array == label)

        # Function to check if points in one array are within threshold of points in another
        def check_within_threshold(points_a, points_b, threshold):
            for point_a in points_a:
                # Broadcasting difference calculation over points_b
                diff = np.abs(points_b - point_a)
                # Check if any point_b is within threshold distance of point_a
                if np.any(np.all(diff <= threshold, axis=1)):
                    return True
            return False

        # Preparing boundary arrays
        north_boundary = labeled_array[:, -1, :]
        south_boundary = labeled_array[:, 0, :]
        east_boundary = labeled_array[:, :, -1]
        west_boundary = labeled_array[:, :, 0]

        # Identifying unique labels on each boundary
        north_labels = np.unique(north_boundary)[1:]  # Excluding background
        south_labels = np.unique(south_boundary)[1:]
        east_labels = np.unique(east_boundary)[1:]
        west_labels = np.unique(west_boundary)[1:]

        # print all booundary labels
        print(f"North labels: {north_labels}")
        print(f"South labels: {south_labels}")
        print(f"East labels: {east_labels}")
        print(f"West labels: {west_labels}")

        # Distance threshold for considering two points as "close"
        threshold = np.array([self.distance_threshold, self.distance_threshold])

        # North-South Merges
        for n_label in north_labels:
            n_points = boundary_points(n_label, north_boundary)
            for s_label in south_labels:
                s_points = boundary_points(s_label, south_boundary)
                if check_within_threshold(n_points, s_points, threshold):
                    merges.append((n_label, s_label))

        # East-West Merges
        for e_label in east_labels:
            e_points = boundary_points(e_label, east_boundary)
            for w_label in west_labels:
                w_points = boundary_points(w_label, west_boundary)
                if check_within_threshold(e_points, w_points, threshold):
                    merges.append((e_label, w_label))


        print(f"Merges: {merges}")
        return merges



    def update_labels_for_merges(self, labeled_array, merges):
        """Update labels for merging regions."""
        print ("Updating labels for merges...")
        for merge_pair in merges:
            # merge_pair is a tuple of two labels to be merged
            min_label = min(merge_pair)
            for label in merge_pair:
                labeled_array[labeled_array == label] = min_label
        return labeled_array

    # @profile
    def create_clouds_from_labeled_array(self, updated_labeled_array, l_data, u_data, v_data, w_data, p_data, theta_l_data, q_t_data, r_data, config, xt, yt, zt):
        """Create Cloud objects from the updated labeled array with optimized performance."""
        print("Creating cloud data from labeled array...")

        # Pre-compute outside temperature for all levels (do this once for all clouds)
        print("Pre-computing environment data...")
        n_levels = len(zt)
        theta_outside_per_level = np.full(n_levels, np.nan)
        
        # Areas outside any cloud (do this calculation once)
        total_mask = updated_labeled_array == 0
        
        # Vectorized calculation of outside temperature
        for z in range(n_levels):
            level_mask = total_mask[z]
            if np.any(level_mask):
                theta_outside_per_level[z] = np.mean(theta_l_data[z][level_mask])

        # Find height index of cloud base
        base_index = np.abs(zt - config['cloud_base_altitude']).argmin()
        
        # Precompute height level indices
        z_to_idx = {z_val: idx for idx, z_val in enumerate(zt)}
        
        # Get the regions
        regions = measure.regionprops(updated_labeled_array, l_data)
        print(f"Processing {len(regions)} regions...")
        
        # Calculate horizontal_resolution_squared once
        horizontal_resolution_squared = config['horizontal_resolution']**2
        
        # Define connectivity structures
        erosion_structure = generate_binary_structure(3, 3)
        ring_structure_2d = generate_binary_structure(2, 1)  # Manhattan (4-neighbour)
        D_ring = int(config.get('env_ring_max_distance', 3))
        periodic_rings = bool(config.get('env_periodic_rings', True))

        # Create Cloud objects in batches to manage memory
        clouds = {}
        batch_size = config.get('cloud_batch_size', 50)

        # --- Fixed Mean Area Sampling Setup ---
        env_aloft_sampling_mode = config.get('env_aloft_sampling_mode', 'exact')
        fixed_sampling_offsets = config.get('fixed_sampling_offsets', None)

        if env_aloft_sampling_mode == 'fixed_mean_area' and fixed_sampling_offsets is None:
            # Calculate mean projected area (in pixels) of all regions
            valid_regions = [r for r in regions if r.area >= config['min_size']]
            if valid_regions:
                projected_areas = []
                for r in valid_regions:
                    # Project along Z axis (axis 0)
                    projected_mask = np.any(r.image, axis=0)
                    projected_areas.append(np.sum(projected_mask))
                
                mean_area_pixels = np.mean(projected_areas)
                
                # Calculate equivalent radius R = sqrt(A / pi)
                R_eq = np.sqrt(mean_area_pixels / np.pi)
                R_ceil = int(np.ceil(R_eq))
                
                # Generate circular offsets
                offsets_y = []
                offsets_x = []
                for dy in range(-R_ceil, R_ceil + 1):
                    for dx in range(-R_ceil, R_ceil + 1):
                        if dy*dy + dx*dx <= R_eq*R_eq:
                            offsets_y.append(dy)
                            offsets_x.append(dx)
                
                fixed_sampling_offsets = (np.array(offsets_y), np.array(offsets_x))
                config['fixed_sampling_offsets'] = fixed_sampling_offsets
                
                print(f"--- Fixed Mean Area Sampling Initialized ---")
                print(f"Mean Projected Area: {mean_area_pixels:.1f} pixels")
                print(f"Equivalent Radius: {R_eq:.2f} pixels")
                print(f"Sample Points per Cloud: {len(offsets_y)}")
            else:
                print("Warning: No valid clouds found. Defaulting to single point.")
                config['fixed_sampling_offsets'] = (np.array([0]), np.array([0]))
        
        for batch_start in range(0, len(regions), batch_size):
            batch_end = min(batch_start + batch_size, len(regions))
            batch_regions = regions[batch_start:batch_end]
            
            for region_idx, region in enumerate(batch_regions):
                if region.area >= config['min_size']:
                    cloud_id = f"{self.timestep}-{region.label}"
                    
                    # Print progress every 50 clouds
                    if (batch_start + region_idx + 1) % 50 == 0:
                        print(f"Processing cloud {batch_start + region_idx + 1} of {len(regions)}")
                    
                    # Get cloud mask and point indices
                    cloud_mask = updated_labeled_array == region.label
                    point_indices = np.argwhere(cloud_mask)
                    
                    # --- Matching Point Calculation (surface + N inward layers) ---
                    # How many inward layers to include for matching (1 = surface only)
                    shell_layers = int(config.get('match_shell_layers', 1))
                    if shell_layers < 1:
                        shell_layers = 1
                    
                    # Build union of shells by successive erosions
                    shell_union = np.zeros_like(cloud_mask, dtype=bool)
                    prev_mask = cloud_mask
                    for _ in range(shell_layers):
                        inner_mask_iter = binary_erosion(prev_mask, structure=erosion_structure)
                        layer_mask = prev_mask & ~inner_mask_iter
                        shell_union |= layer_mask # 
                        prev_mask = inner_mask_iter
                        if not np.any(prev_mask):
                            break
                    
                    # These are the points used for matching (augmented "surface")
                    surface_point_indices = np.argwhere(shell_union)

                    # Skip if no points (shouldn't happen but just in case)
                    if len(point_indices) == 0:
                        continue
                    
                    # Create points array
                    points = np.column_stack([
                        xt[point_indices[:, 2]],
                        yt[point_indices[:, 1]],
                        zt[point_indices[:, 0]]
                    ])
                    
                    surface_points = np.column_stack([
                        xt[surface_point_indices[:, 2]],
                        yt[surface_point_indices[:, 1]],
                        zt[surface_point_indices[:, 0]]
                    ])

                    # Extract data using vectorized operations
                    u_values = u_data[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]]
                    v_values = v_data[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]]
                    w_values = w_data[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]]
                    l_values = l_data[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]]
                    p_values = p_data[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]]
                    theta_l_values = theta_l_data[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]]
                    # Note: RICO data already in kg/kg (metadata labels g/kg incorrectly)
                    q_t_values = q_t_data[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]]
                    r_values = r_data[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]]
                    q_l_values = l_values
                    q_v_values = q_t_values - q_l_values

                    # Convert any masked arrays to regular arrays (needed for Numba)
                    if hasattr(p_values, 'filled'):
                        p_values = p_values.filled(np.nan)
                    if hasattr(theta_l_values, 'filled'):
                        theta_l_values = theta_l_values.filled(np.nan)
                    if hasattr(q_l_values, 'filled'):
                        q_l_values = q_l_values.filled(np.nan)
                    if hasattr(q_v_values, 'filled'):
                        q_v_values = q_v_values.filled(np.nan)
                    if hasattr(r_values, 'filled'):
                        r_values = r_values.filled(np.nan)
                    if hasattr(w_values, 'filled'):
                        w_values = w_values.filled(np.nan)
                    
                    # --- Calculate Per-Cloud Mean Velocities ---
                    mean_u = np.mean(u_values)
                    mean_v = np.mean(v_values)
                    mean_w = np.mean(w_values)

                    # Calculate max_w
                    max_w = np.max(w_values)
                    
                    # Get cloud base information using vectorized operations
                    base_mask = point_indices[:, 0] == base_index
                    if np.any(base_mask):
                        base_w_values = w_values[base_mask]
                        max_w_cloud_base = np.max(base_w_values)
                        cloud_base_area = np.sum(base_mask) * horizontal_resolution_squared
                    else:
                        max_w_cloud_base = np.nan
                        cloud_base_area = 0
                        
                    # Calculate max height with vectorized operations
                    max_height = np.max(zt[point_indices[:, 0]])
                    
                    # Calculate min height with vectorized operations
                    min_height = np.min(zt[point_indices[:, 0]])
                    
                    # Compute surface area
                    surface_mask = binary_dilation(cloud_mask) & ~cloud_mask
                    surface_area = np.sum(surface_mask) * config['horizontal_resolution']**2
                    
                    # Calculate ql flux using vectorized operations
                    ql_flux = np.sum(w_values * l_values)
                    
                    # Use Numba to accelerate physics calculations
                    # Rain is excluded from saturation adjustment but included in density loading
                    temps, rhos, mass_fluxes = calculate_physics_variables(
                        p_values, theta_l_values, q_l_values, q_v_values, 
                        w_values, horizontal_resolution_squared,
                        r_rain_values=r_values
                    )
                    
                    # Pre-allocate arrays for per-level data
                    mass_flux_per_level = np.full(n_levels, np.nan)
                    temp_per_level = np.full(n_levels, np.nan)
                    w_per_level = np.full(n_levels, np.nan)
                    u_per_level = np.full(n_levels, np.nan)
                    v_per_level = np.full(n_levels, np.nan)
                    circum_per_level = np.full(n_levels, np.nan)
                    eff_radius_per_level = np.full(n_levels, np.nan)  # legacy compactness ratio
                    area_per_level = np.full(n_levels, np.nan)
                    equiv_radius_per_level = np.full(n_levels, np.nan)
                    compactness_per_level = np.full(n_levels, np.nan)
                    
                    # Get unique z levels and their counts
                    unique_z_levels, z_level_counts = np.unique(point_indices[:, 0], return_counts=True)
                    
                    # Process data by level using vectorized operations
                    for z_level_idx, count in zip(unique_z_levels, z_level_counts):
                        # Get points at this level
                        level_mask = point_indices[:, 0] == z_level_idx
                        
                        # Get values for this level
                        level_w_values = w_values[level_mask]
                        level_temps = temps[level_mask]
                        level_mass_fluxes = mass_fluxes[level_mask]
                        level_u_values = u_values[level_mask]
                        level_v_values = v_values[level_mask]
                        
                        # Calculate level statistics
                        w_per_level[z_level_idx] = np.mean(level_w_values)
                        u_per_level[z_level_idx] = np.mean(level_u_values)
                        v_per_level[z_level_idx] = np.mean(level_v_values)
                        temp_per_level[z_level_idx] = np.mean(level_temps)
                        mass_flux_per_level[z_level_idx] = np.sum(level_mass_fluxes)
                        
                        # Calculate circumference and effective radius
                        cloud_slice = updated_labeled_array[z_level_idx, :, :] == region.label
                        if np.any(cloud_slice):
                            # perimeter in grid cells -> meters
                            perimeter_cells = measure.perimeter(cloud_slice, neighborhood=8)
                            perimeter = perimeter_cells * config['horizontal_resolution']
                            circum_per_level[z_level_idx] = perimeter
                            # Area [m2]
                            area = np.sum(cloud_slice) * horizontal_resolution_squared
                            area_per_level[z_level_idx] = area
                            if area > 0:
                                r_eq = np.sqrt(area / np.pi)
                                equiv_radius_per_level[z_level_idx] = r_eq
                                # compactness >=1
                                comp = perimeter / (2.0 * np.pi * r_eq)
                                compactness_per_level[z_level_idx] = comp
                                eff_radius_per_level[z_level_idx] = comp  # legacy slot
                    
                    # Calculate total mass flux
                    mass_flux = np.nansum(mass_flux_per_level)
                    
                    # Diagnosed base radius
                    base_radius_diagnosed = np.nan
                    base_area_diagnosed = np.nan
                    # Lowest occupied vertical index
                    min_z_idx = int(np.min(point_indices[:, 0]))
                    r0 = equiv_radius_per_level[min_z_idx]
                    if not np.isnan(r0):
                        threshold = config.get('base_increase_threshold', 1.5)
                        scan_levels = config.get('base_scan_levels', 3)
                        chosen_r = r0
                        scan_limit = min_z_idx + 1 + scan_levels
                        for zscan in range(min_z_idx + 1, scan_limit):
                            rscan = equiv_radius_per_level[zscan]
                            if not np.isnan(rscan) and rscan >= threshold * r0:
                                chosen_r = rscan
                                break
                        base_radius_diagnosed = chosen_r
                        base_area_diagnosed = np.pi * chosen_r * chosen_r
                    # Max equivalent radius
                    valid_r = equiv_radius_per_level[~np.isnan(equiv_radius_per_level)]
                    max_equiv_radius = np.nan if valid_r.size == 0 else np.max(valid_r)
                    
                    # Create points array
                    points = np.column_stack([
                        xt[point_indices[:, 2]],
                        yt[point_indices[:, 1]],
                        zt[point_indices[:, 0]]
                    ])

                    # Physical centroid in (x,y,z)
                    phys_centroid = points.mean(axis=0)

                    # --- Environment Aloft Analysis ---
                    # Analyze environment above the cloud top
                    n_aloft_levels = int(config.get('env_aloft_levels', 40))
                    if n_aloft_levels == -1:
                        n_aloft_levels = n_levels
                    
                    # Initialize arrays for profiles
                    env_aloft_qt_diff = np.full(n_aloft_levels, np.nan)
                    env_aloft_thetal_diff = np.full(n_aloft_levels, np.nan)
                    env_aloft_shear = np.full(n_aloft_levels, np.nan)
                    env_aloft_n2 = np.full(n_aloft_levels, np.nan)
                    env_aloft_rh = np.full(n_aloft_levels, np.nan)

                    # Initialize arrays for variability (std)
                    env_aloft_qt_std = np.full(n_aloft_levels, np.nan)
                    env_aloft_thetal_std = np.full(n_aloft_levels, np.nan)
                    env_aloft_shear_std = np.full(n_aloft_levels, np.nan)
                    env_aloft_n2_std = np.full(n_aloft_levels, np.nan)
                    env_aloft_rh_std = np.full(n_aloft_levels, np.nan)
                    
                    # Find cloud top for each column using region.image (efficiently)
                    # region.image is (Z, Y, X) boolean mask in the bounding box
                    # We want max Z for each (Y, X) where cloud is present
                    # Create an array of Z indices matching the image shape
                    z_indices_local = np.arange(region.image.shape[0]).reshape(-1, 1, 1)
                    # Mask where cloud is not present with -1
                    masked_z = np.where(region.image, z_indices_local, -1)
                    # Find max Z along axis 0 (vertical)
                    top_z_local = np.max(masked_z, axis=0) # Shape (Y_box, X_box)
                    
                    # Filter valid columns (where top_z_local != -1)
                    valid_mask = top_z_local != -1
                    valid_ys, valid_xs = np.where(valid_mask)
                    valid_z_locals = top_z_local[valid_ys, valid_xs]

                    # --- Environment Aloft Mode ---
                    # 'flat': Use the maximum height of the cloud for all columns (flat plane above max top)
                    # 'local': Use the local cloud top height for each column (following cloud terrain)
                    env_aloft_mode = config.get('env_aloft_mode', 'flat')
                    if env_aloft_mode == 'flat':
                        # Use the maximum height of the cloud for all columns
                        # valid_z_locals contains local Z indices relative to bbox min_z
                        if len(valid_z_locals) > 0:
                            max_local_z = np.max(valid_z_locals)
                            valid_z_locals[:] = max_local_z
                    
                    # Convert to absolute coordinates
                    # region.bbox is (min_z, min_y, min_x, max_z, max_y, max_x)
                    abs_z_tops = valid_z_locals + region.bbox[0]
                    abs_ys = valid_ys + region.bbox[1]
                    abs_xs = valid_xs + region.bbox[2]

                    # --- Sampling Mode Logic ---
                    if env_aloft_sampling_mode == 'fixed_mean_area' and fixed_sampling_offsets is not None:
                        if env_aloft_mode != 'flat':
                            raise ValueError("Error: 'fixed_mean_area' sampling requires 'env_aloft_mode' to be 'flat'.")
                        
                        # Calculate Centroid (Y, X) in absolute coordinates
                        local_centroid = region.centroid
                        centroid_y_idx = int(local_centroid[1] + region.bbox[1])
                        centroid_x_idx = int(local_centroid[2] + region.bbox[2])
                        
                        # Apply pre-calculated circular offsets
                        offsets_y, offsets_x = fixed_sampling_offsets
                        
                        # Wrap periodic boundaries
                        ny = len(yt)
                        nx = len(xt)
                        sample_ys = (centroid_y_idx + offsets_y) % ny
                        sample_xs = (centroid_x_idx + offsets_x) % nx
                        
                        # Update abs_ys, abs_xs
                        abs_ys = sample_ys
                        abs_xs = sample_xs
                        
                        # Update abs_z_tops (all flat at max height)
                        if len(abs_z_tops) > 0:
                            max_z_idx = np.max(abs_z_tops)
                        else:
                            max_z_idx = int(max_height / (zt[1]-zt[0]))
                        
                        abs_z_tops = np.full(len(abs_ys), max_z_idx)

                    
                    # Loop over levels aloft
                    for k in range(n_aloft_levels):
                        # Target z indices for all columns (k=0 is 1 level above top)
                        target_zs = abs_z_tops + k + 1 
                        
                        # Filter out of bounds
                        in_bounds = target_zs < n_levels
                        if not np.any(in_bounds):
                            continue
                            
                        # Select valid points
                        z_sel = target_zs[in_bounds]
                        y_sel = abs_ys[in_bounds]
                        x_sel = abs_xs[in_bounds]
                        
                        # Extract data (RICO data already in kg/kg)
                        qt_vals = q_t_data[z_sel, y_sel, x_sel]
                        thetal_vals = theta_l_data[z_sel, y_sel, x_sel]
                        p_vals = p_data[z_sel, y_sel, x_sel]
                        
                        # Convert masked arrays to numpy arrays with NaNs for Numba compatibility
                        qt_vals = np.ma.filled(qt_vals, np.nan)
                        thetal_vals = np.ma.filled(thetal_vals, np.nan)
                        p_vals = np.ma.filled(p_vals, np.nan)
                        
                        # 1. Anomalies (mean_qt_profile also in kg/kg)
                        mean_qt_vals = self.mean_qt_profile[z_sel]
                        mean_thetal_vals = self.mean_theta_l_profile[z_sel]
                        
                        env_aloft_qt_diff[k] = np.nanmean(qt_vals - mean_qt_vals)
                        env_aloft_qt_std[k] = np.nanstd(qt_vals)
                        env_aloft_thetal_diff[k] = np.nanmean(thetal_vals - mean_thetal_vals)
                        env_aloft_thetal_std[k] = np.nanstd(thetal_vals)
                        
                        # 2. Shear
                        # Need u, v at z+1 and z-1 (or z and z+1)
                        # Check bounds for neighbors
                        z_up = np.minimum(z_sel + 1, n_levels - 1)
                        z_down = np.maximum(z_sel - 1, 0)
                        dz_local = zt[z_up] - zt[z_down] # Array of dz
                        
                        # Avoid division by zero if z_up == z_down (single level domain?)
                        # Should not happen if n_levels > 1
                        
                        u_up = u_data[z_up, y_sel, x_sel]
                        u_down = u_data[z_down, y_sel, x_sel]
                        v_up = v_data[z_up, y_sel, x_sel]
                        v_down = v_data[z_down, y_sel, x_sel]
                        
                        du_dz = (u_up - u_down) / dz_local
                        dv_dz = (v_up - v_down) / dz_local
                        shear_vals = np.sqrt(du_dz**2 + dv_dz**2)
                        env_aloft_shear[k] = np.nanmean(shear_vals)
                        env_aloft_shear_std[k] = np.nanstd(shear_vals)
                        
                        # 3. N^2
                        # theta_v = theta_l * (1 + 0.61 * qt) where qt is in kg/kg
                        
                        tl_up = theta_l_data[z_up, y_sel, x_sel]
                        qt_up = q_t_data[z_up, y_sel, x_sel]  # already kg/kg
                        tv_up = tl_up * (1 + 0.61 * qt_up)
                        
                        tl_down = theta_l_data[z_down, y_sel, x_sel]
                        qt_down = q_t_data[z_down, y_sel, x_sel]  # already kg/kg
                        tv_down = tl_down * (1 + 0.61 * qt_down)
                        
                        tl_curr = thetal_vals
                        qt_curr = qt_vals
                        tv_curr = tl_curr * (1 + 0.61 * qt_curr)
                        
                        dtv_dz = (tv_up - tv_down) / dz_local
                        n2_vals = (const.g / tv_curr) * dtv_dz
                        env_aloft_n2[k] = np.nanmean(n2_vals)
                        env_aloft_n2_std[k] = np.nanstd(n2_vals)
                        
                        # 4. RH
                        _, rh_vals = calculate_rh_and_temperature(p_vals, thetal_vals, qt_vals)
                        env_aloft_rh[k] = np.nanmean(rh_vals)
                        env_aloft_rh_std[k] = np.nanstd(rh_vals)

                    # Create a Cloud object and store it
                    clouds[cloud_id] = Cloud(
                        cloud_id=cloud_id,
                        size=region.area,
                        surface_area=surface_area,
                        location=(phys_centroid[0], phys_centroid[1], phys_centroid[2]),
                        points=points,
                        surface_points=surface_points,
                        max_height=max_height,
                        max_w=max_w,
                        max_w_cloud_base=max_w_cloud_base,
                        mean_u=mean_u,
                        mean_v=mean_v,
                        mean_w=mean_w,
                        cloud_base_area=cloud_base_area,
                        ql_flux=ql_flux,
                        mass_flux=mass_flux,
                        mass_flux_per_level=mass_flux_per_level,
                        temp_per_level=temp_per_level,
                        theta_outside_per_level=theta_outside_per_level,
                        w_per_level=w_per_level,
                        circum_per_level=circum_per_level,
                        eff_radius_per_level=eff_radius_per_level,
                        timestep=self.timestep,
                        cloud_base_height=min_height,
                        area_per_level=area_per_level,
                        equiv_radius_per_level=equiv_radius_per_level,
                        compactness_per_level=compactness_per_level,
                        base_radius_diagnosed=base_radius_diagnosed,
                        base_area_diagnosed=base_area_diagnosed,
                        max_equiv_radius=max_equiv_radius,
                        # NIP kinematics per level
                        u_per_level=u_per_level,
                        v_per_level=v_per_level,
                        # Environment Aloft
                        env_aloft_qt_diff=env_aloft_qt_diff,
                        env_aloft_thetal_diff=env_aloft_thetal_diff,
                        env_aloft_shear=env_aloft_shear,
                        env_aloft_n2=env_aloft_n2,
                        env_aloft_rh=env_aloft_rh,
                        env_aloft_qt_std=env_aloft_qt_std,
                        env_aloft_thetal_std=env_aloft_thetal_std,
                        env_aloft_shear_std=env_aloft_shear_std,
                        env_aloft_n2_std=env_aloft_n2_std,
                        env_aloft_rh_std=env_aloft_rh_std
                    )

                    # --- Environment ring averages (per level, per ring distance) ---
                    # Initialize arrays with NaN
                    n_rings = D_ring
                    env_w_r = np.full((n_levels, n_rings), np.nan, dtype=float)
                    env_l_r = np.full((n_levels, n_rings), np.nan, dtype=float)
                    env_qt_r = np.full((n_levels, n_rings), np.nan, dtype=float)
                    env_qv_r = np.full((n_levels, n_rings), np.nan, dtype=float)
                    env_p_r = np.full((n_levels, n_rings), np.nan, dtype=float)
                    env_theta_r = np.full((n_levels, n_rings), np.nan, dtype=float)
                    env_buoy_r = np.full((n_levels, n_rings), np.nan, dtype=float)

                    # Precompute: baseline environment mean density per level
                    rho_env_mean = getattr(self, 'env_rho_mean_per_level', None)

                    # For each occupied level, compute rings
                    for z_level_idx in unique_z_levels:
                        cloud_slice = updated_labeled_array[z_level_idx, :, :] == region.label
                        if not np.any(cloud_slice):
                            continue
                        env_slice = (updated_labeled_array[z_level_idx, :, :] == 0)

                        # Prepare periodic padding (or constant)
                        padw = ((D_ring, D_ring), (D_ring, D_ring))
                        if periodic_rings:
                            cloud_pad = np.pad(cloud_slice, padw, mode='wrap')
                            env_pad = np.pad(env_slice, padw, mode='wrap')
                        else:
                            cloud_pad = np.pad(cloud_slice, padw, mode='constant', constant_values=False)
                            env_pad = np.pad(env_slice, padw, mode='constant', constant_values=False)

                        # Compute successive dilations on padded grid
                        # Iter 0 is the cloud itself (for convenience)
                        prev = cloud_pad
                        for d in range(1, D_ring + 1):
                            dil = binary_dilation(cloud_pad, structure=ring_structure_2d, iterations=d)
                            ring_pad = dil & (~prev)
                            # Crop back to original slice
                            ring_center = ring_pad[D_ring:-D_ring, D_ring:-D_ring]
                            # Keep only environment points
                            ring_env = ring_center & env_slice
                            if np.any(ring_env):
                                # Extract ring values
                                w_z = w_data[z_level_idx]
                                l_z = l_data[z_level_idx]
                                p_z = p_data[z_level_idx]
                                t_z = theta_l_data[z_level_idx]
                                qt_z = q_t_data[z_level_idx]
                                r_z = r_data[z_level_idx]

                                # Handle masked arrays
                                if hasattr(w_z, 'filled'):
                                    w_z = w_z.filled(np.nan)
                                if hasattr(l_z, 'filled'):
                                    l_z = l_z.filled(np.nan)
                                if hasattr(p_z, 'filled'):
                                    p_z = p_z.filled(np.nan)
                                if hasattr(t_z, 'filled'):
                                    t_z = t_z.filled(np.nan)
                                if hasattr(qt_z, 'filled'):
                                    qt_z = qt_z.filled(np.nan)
                                if hasattr(r_z, 'filled'):
                                    r_z = r_z.filled(np.nan)

                                # Compute qv in same units as inputs for output; also kg/kg for physics
                                qv_z = qt_z - l_z

                                # Means (environment ring)
                                j = d - 1
                                env_w_r[z_level_idx, j] = np.nanmean(w_z[ring_env])
                                env_l_r[z_level_idx, j] = np.nanmean(l_z[ring_env])
                                env_qt_r[z_level_idx, j] = np.nanmean(qt_z[ring_env])
                                env_qv_r[z_level_idx, j] = np.nanmean(qv_z[ring_env])
                                env_p_r[z_level_idx, j] = np.nanmean(p_z[ring_env])
                                env_theta_r[z_level_idx, j] = np.nanmean(t_z[ring_env])

                                # Buoyancy: compute densities for ring points
                                # Note: RICO data already in kg/kg
                                ql_ring = l_z[ring_env]
                                qt_ring = qt_z[ring_env]
                                qv_ring = qt_ring - ql_ring
                                p_ring = p_z[ring_env]
                                t_ring = t_z[ring_env]
                                w_ring = w_z[ring_env]
                                r_ring = r_z[ring_env]
                                _, rhos_ring, _ = calculate_physics_variables(
                                    p_ring, t_ring, ql_ring, qv_ring, w_ring,
                                    horizontal_resolution_squared,
                                    r_rain_values=r_ring
                                )
                                rho0 = rho_env_mean[z_level_idx] if rho_env_mean is not None else np.nan
                                if np.isfinite(rho0) and np.any(np.isfinite(rhos_ring)):
                                    g = 9.81
                                    with np.errstate(invalid='ignore', divide='ignore'):
                                        b_ring = -g * (rhos_ring - rho0) / rho0
                                    env_buoy_r[z_level_idx, j] = np.nanmean(b_ring)

                            prev = dil

                    # Attach to cloud
                    clouds[cloud_id].env_w_rings = env_w_r
                    clouds[cloud_id].env_l_rings = env_l_r
                    clouds[cloud_id].env_qt_rings = env_qt_r
                    clouds[cloud_id].env_qv_rings = env_qv_r
                    clouds[cloud_id].env_p_rings = env_p_r
                    clouds[cloud_id].env_theta_l_rings = env_theta_r
                    clouds[cloud_id].env_buoyancy_rings = env_buoy_r
            
            # Force garbage collection after each batch
            gc.collect()
        
        print(f"Cloud data for {len(clouds)} objects.")
        return clouds

    def build_global_surface_kdtree(self):
        """Build a single KD-tree for all surface points in the cloud field."""
        if not self.clouds:
            return
            
        print("Building global surface point KD-tree...")
        
        # Calculate total number of surface points
        total_points = sum(cloud.surface_points.shape[0] for cloud in self.clouds.values())
        
        # Pre-allocate arrays for better memory efficiency
        self.surface_points_array = np.empty((total_points, 3), dtype=np.float32)
        self.surface_point_to_cloud_id = np.empty(total_points, dtype=object)
        
        # Fill arrays with surface points and their cloud IDs
        idx = 0
        for cloud_id, cloud in self.clouds.items():
            n_points = cloud.surface_points.shape[0]
            if n_points > 0:
                self.surface_points_array[idx:idx+n_points] = cloud.surface_points
                self.surface_point_to_cloud_id[idx:idx+n_points] = cloud_id
                idx += n_points
        
        # Trim arrays to actual size (in case some clouds had no surface points)
        if idx < total_points:
            self.surface_points_array = self.surface_points_array[:idx]
            self.surface_point_to_cloud_id = self.surface_point_to_cloud_id[:idx]
        
        # Build the KD-tree using only X,Y coordinates
        if len(self.surface_points_array) > 0:
            self.surface_points_kdtree = cKDTree(self.surface_points_array[:, :2])
            print(f"KD-tree built with {len(self.surface_points_array)} surface points")


    # NIP: Neighbour Interaction Potential
    def _compute_nip(self, config):
        """
        Computes instantaneous per-level NIP for every cloud and store field-level scales
        required for temporal accumulation (T per level, Lh, Vref per level).

        Notes:
        - Uses periodic minimum-image distance in x,y using domain sizes from xt, yt.
        - Per-level winds are area-mean over in-cloud points.
        - Normalizes by median mass flux per level and pi*Lh^2.
        - Guards against zero/NaN medians and empty neighbours.
        """
        if not self.clouds:
            self.nip_Lh = np.nan
            self.nip_Vref_per_level = None
            self.nip_T_per_level = None
            return


        clouds = self.clouds
        cloud_ids = list(clouds.keys())
        n_clouds = len(cloud_ids)
        print(f"Calculating NIP for timestep {self.timestep}: {n_clouds} clouds")
        zt = self.zt
        n_levels = len(zt)

        # Domain sizes (periodic)
        dx = config.get('horizontal_resolution', float(self.xt[1]-self.xt[0]))
        Lx = (self.xt[-1] - self.xt[0]) + dx
        Ly = (self.yt[-1] - self.yt[0]) + dx

        # Parameters
        gamma = float(config.get('nip_gamma', 0.3))
        f_radius = float(config.get('nip_f', 3.0))
        Lh_min = float(config.get('nip_Lh_min', 100.0))
        Lh_max = float(config.get('nip_Lh_max', 2000.0))
        T_min = float(config.get('nip_T_min', 60.0))
        T_max = float(config.get('nip_T_max', 1800.0))
        Vref_floor = float(config.get('nip_Vref_floor', 1.0))

        # Gather centroids for distance calculations
        centroids_xy = np.array([[clouds[cid].location[0], clouds[cid].location[1]] for cid in cloud_ids], dtype=float)

        # Pairwise periodic distances (minimum-image convention)
        def min_image_delta(a, b, L):
            d = (b - a + 0.5 * L) % L - 0.5 * L
            return d

        # Compute nearest-neighbour distances for Lh
        nn_dists = np.empty(n_clouds, dtype=float)
        for i in range(n_clouds):
            xi, yi = centroids_xy[i]
            best = np.inf
            for j in range(n_clouds):
                if i == j:
                    continue
                xj, yj = centroids_xy[j]
                dxij = min_image_delta(xi, xj, Lx)
                dyij = min_image_delta(yi, yj, Ly)
                dij = np.hypot(dxij, dyij)
                if dij < best:
                    best = dij
            nn_dists[i] = best if np.isfinite(best) else np.nan

        # Lh = 2 * median(NND), clipped
        valid_nnd = nn_dists[np.isfinite(nn_dists) & (nn_dists > 0)]
        if valid_nnd.size == 0:
            Lh = Lh_min
        else:
            Lh = 2.0 * float(np.median(valid_nnd))
            Lh = float(np.clip(Lh, Lh_min, Lh_max))

        # Global safety search radius
        R_global = min(3.0 * Lh, 0.5 * min(Lx, Ly))

        # Compute per-level median mass flux across clouds and per-level Vref
        # Also ensure per-cloud per-level u/v arrays exist; if not, approximate using cloud mean
        mass_flux_stack = np.full((n_clouds, n_levels), np.nan, dtype=float)
        u_stack = np.full((n_clouds, n_levels), np.nan, dtype=float)
        v_stack = np.full((n_clouds, n_levels), np.nan, dtype=float)
        r_eq_stack = np.full((n_clouds, n_levels), np.nan, dtype=float)
        for idx, cid in enumerate(cloud_ids):
            c = clouds[cid]
            mass_flux_stack[idx, :] = c.mass_flux_per_level
            # Kinematics per level; if missing, fill with overall mean
            if getattr(c, 'u_per_level', None) is None or getattr(c, 'v_per_level', None) is None:
                u_stack[idx, :] = np.full(n_levels, c.mean_u)
                v_stack[idx, :] = np.full(n_levels, c.mean_v)
            else:
                u_stack[idx, :] = c.u_per_level
                v_stack[idx, :] = c.v_per_level
            r_eq_stack[idx, :] = c.equiv_radius_per_level

        # Median mass flux per level across clouds; use positive values when possible
        med_mass_flux = np.full(n_levels, np.nan, dtype=float)
        for z in range(n_levels):
            mfz = mass_flux_stack[:, z]
            # prefer positive-only median; fallback to all finite
            pos = mfz[np.isfinite(mfz) & (mfz > 0)]
            if pos.size > 0:
                med_mass_flux[z] = float(np.median(pos))
            else:
                fin = mfz[np.isfinite(mfz)]
                med_mass_flux[z] = float(np.median(fin)) if fin.size > 0 else np.nan

        # Vref per level: median speed across clouds, only where level is occupied
        Vref = np.full(n_levels, np.nan, dtype=float)
        for z in range(n_levels):
            speeds = np.hypot(u_stack[:, z], v_stack[:, z])
            speeds = speeds[np.isfinite(speeds)]
            # If no clouds occupy this level (no finite speeds), leave Vref[z] = NaN
            if speeds.size > 0:
                Vref[z] = float(np.median(speeds))

        # Time scale per level
        T = np.full(n_levels, np.nan, dtype=float)
        for z in range(n_levels):
            # Only define T where Vref is finite and positive; else leave NaN
            if np.isfinite(Vref[z]) and Vref[z] > 0:
                T[z] = float(np.clip(Lh / Vref[z], T_min, T_max))

        # One-line summary
        Vref_med = float(np.nanmedian(Vref)) if np.any(np.isfinite(Vref)) else float('nan')
        T_med = float(np.nanmedian(T)) if np.any(np.isfinite(T)) else float('nan')
        print(f"NIP scales: Lh={Lh:.1f} m, Vref_med={Vref_med:.2f} m/s, T_med={T_med:.1f} s")

        # Compute NIP per level for each cloud
        Dmax = ( (0.5*Lx)**2 + (0.5*Ly)**2 )**0.5
        N_norm = np.pi * (Lh**2)  # part not including mass-flux median

        for i, cid in enumerate(cloud_ids):
            ci = clouds[cid]
            xi, yi = centroids_xy[i]
            nip_i = np.zeros(n_levels, dtype=float)

            # For each neighbour j
            for j, cjd in enumerate(cloud_ids):
                if j == i:
                    continue
                xj, yj = centroids_xy[j]
                dxij = min_image_delta(xi, xj, Lx)
                dyij = min_image_delta(yi, yj, Ly)
                dij = float(np.hypot(dxij, dyij))
                if dij <= 0.0:
                    continue
                # unit vector from i->j
                rx = dxij / dij
                ry = dyij / dij

                cj = clouds[cjd]
                # For each level, check search radius and accumulate
                for z in range(n_levels):
                    r_eq_i = r_eq_stack[i, z]
                    if not np.isfinite(r_eq_i) or r_eq_i <= 0:
                        continue
                    # Dynamic search radius clipped by global safety
                    Ri = min(f_radius * r_eq_i, R_global, Dmax)
                    if dij > Ri:
                        continue

                    mf_med = med_mass_flux[z]
                    if not np.isfinite(mf_med) or mf_med <= 0:
                        # no meaningful normalizer/mark for this level
                        continue
                    Mj = mass_flux_stack[j, z]
                    if not np.isfinite(Mj) or Mj <= 0:
                        continue
                    mark = Mj / mf_med

                    # Kinematic boost
                    ui = u_stack[i, z]; vi = v_stack[i, z]
                    uj = u_stack[j, z]; vj = v_stack[j, z]
                    if not (np.isfinite(ui) and np.isfinite(vi) and np.isfinite(uj) and np.isfinite(vj)):
                        kin_boost = 1.0
                    else:
                        vin = -((uj - ui) * rx + (vj - vi) * ry)  # positive when j approaches i
                        if np.isfinite(Vref[z]) and Vref[z] > 0:
                            kin_boost = 1.0 + gamma * max(0.0, float(vin)) / float(Vref[z])
                        else:
                            kin_boost = 1.0

                    kernel = np.exp(-dij / Lh)
                    nip_i[z] += mark * kernel * kin_boost

            # Final normalization: divide by π Lh^2 only (marks already normalized by mf_med)
            for z in range(n_levels):
                ci_nip = nip_i[z] / N_norm if N_norm > 0 else 0.0
                if ci.nip_per_level is None:
                    ci.nip_per_level = np.full(n_levels, np.nan, dtype=float)
                ci.nip_per_level[z] = ci_nip

        # Store field-level scales for accumulation use
        self.nip_Lh = Lh
        self.nip_Vref_per_level = Vref
        self.nip_T_per_level = T
