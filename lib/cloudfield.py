import numpy as np
import gc
from skimage import measure
from memory_profiler import profile
from scipy.ndimage import binary_dilation  # Keep this for other operations
# Remove: from scipy.ndimage import binary_erosion, generate_binary_structure
from skimage.morphology import binary_erosion, cube  # Add these imports
from utils.plotting_utils import plot_labeled_regions
from lib.cloud import Cloud
from utils.physics import calculate_physics_variables
from scipy.spatial import cKDTree

class CloudField:
    """Class to identify and track clouds in a labeled data field."""
    def __init__(self, l_data, u_data, v_data, w_data, p_data, theta_l_data, q_t_data, timestep, config, xt, yt, zt):
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
            updated_labeled_array, p_data, theta_l_data, l_data, q_t_data, w_data, config
        )

        # create cloud data from updated labeled array
        self.clouds = self.create_clouds_from_labeled_array(
            updated_labeled_array, l_data, u_data, v_data, w_data, p_data, theta_l_data, q_t_data, config, xt, yt, zt)

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

    def _calculate_environment_mass_flux(self, labeled_array, p_data, theta_l_data, l_data, q_t_data, w_data, config):
        """Calculate mass flux for the environment (non-cloud regions) at each level."""
        print("Calculating environment mass flux...")
        env_mask = labeled_array == 0
        n_levels = len(self.zt)
        
        if not np.any(env_mask):
            return np.zeros(n_levels)

        # Get coordinates of all environment points
        z_indices, _, _ = np.where(env_mask)

        # Extract 1D arrays of physical variables for all environment points
        p_values_env = p_data[env_mask]
        theta_l_values_env = theta_l_data[env_mask]
        l_values_env = l_data[env_mask]
        q_t_values_env = q_t_data[env_mask]
        w_values_env = w_data[env_mask]

        # Convert units for physics calculation
        q_l_values_env = l_values_env / 1000.0
        q_t_values_env_kg = q_t_values_env / 1000.0
        q_v_values_env = q_t_values_env_kg - q_l_values_env

        # Convert any masked arrays to regular arrays (needed for Numba)
        if hasattr(p_values_env, 'filled'):
            p_values_env = p_values_env.filled(np.nan)
        if hasattr(theta_l_values_env, 'filled'):
            theta_l_values_env = theta_l_values_env.filled(np.nan)
        if hasattr(q_l_values_env, 'filled'):
            q_l_values_env = q_l_values_env.filled(np.nan)
        if hasattr(q_v_values_env, 'filled'):
            q_v_values_env = q_v_values_env.filled(np.nan)
        if hasattr(w_values_env, 'filled'):
            w_values_env = w_values_env.filled(np.nan)

        # Reuse the existing physics function to get mass flux for each point
        _, _, mass_fluxes_env = calculate_physics_variables(
            p_values_env,
            theta_l_values_env,
            q_l_values_env,
            q_v_values_env,
            w_values_env,
            config['horizontal_resolution']**2
        )

        # Sum the mass fluxes per level using the z-indices
        # np.bincount is highly efficient for this aggregation task
        env_mass_flux_per_level = np.bincount(z_indices, weights=mass_fluxes_env, minlength=n_levels)
        
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
    def create_clouds_from_labeled_array(self, updated_labeled_array, l_data, u_data, v_data, w_data, p_data, theta_l_data, q_t_data, config, xt, yt, zt):
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
        
        # Create a 3×3×3 cube for 26-connectivity erosion operations
        erosion_footprint = cube(3)

        # Create Cloud objects in batches to manage memory
        clouds = {}
        batch_size = 50  # Process 50 clouds at a time
        
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
                    
                    # --- Surface Point Calculation ---
                    # Erode the mask to find the "inner" points using skimage's optimized implementation
                    inner_mask = binary_erosion(cloud_mask, footprint=erosion_footprint)
                    # The surface is the original mask minus the inner part.
                    surface_mask = cloud_mask & ~inner_mask
                    surface_point_indices = np.argwhere(surface_mask)

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
                    q_t_values = q_t_data[point_indices[:, 0], point_indices[:, 1], point_indices[:, 2]] / 1000  # g/kg to kg/kg
                    q_l_values = l_values / 1000  # g/kg to kg/kg
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
                    temps, rhos, mass_fluxes = calculate_physics_variables(
                        p_values, theta_l_values, q_l_values, q_v_values, 
                        w_values, horizontal_resolution_squared
                    )
                    
                    # Pre-allocate arrays for per-level data
                    mass_flux_per_level = np.full(n_levels, np.nan)
                    temp_per_level = np.full(n_levels, np.nan)
                    w_per_level = np.full(n_levels, np.nan)
                    circum_per_level = np.full(n_levels, np.nan)
                    eff_radius_per_level = np.full(n_levels, np.nan)
                    
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
                        
                        # Calculate level statistics
                        w_per_level[z_level_idx] = np.mean(level_w_values)
                        temp_per_level[z_level_idx] = np.mean(level_temps)
                        mass_flux_per_level[z_level_idx] = np.sum(level_mass_fluxes)
                        
                        # Calculate circumference and effective radius
                        cloud_slice = updated_labeled_array[z_level_idx, :, :] == region.label
                        if np.any(cloud_slice):
                            perimeter = measure.perimeter(cloud_slice, neighborhood=8)
                            circum_per_level[z_level_idx] = perimeter * config['horizontal_resolution']
                            
                            area = np.sum(cloud_slice) * horizontal_resolution_squared
                            circle_circumference = 2 * np.pi * np.sqrt(area / np.pi)
                            eff_radius_per_level[z_level_idx] = circum_per_level[z_level_idx] / circle_circumference
                    
                    # Calculate total mass flux
                    mass_flux = np.nansum(mass_flux_per_level)
                    
                    # Create a Cloud object and store it
                    clouds[cloud_id] = Cloud(
                        cloud_id=cloud_id,
                        size=region.area,
                        surface_area=surface_area,
                        location=(xt[int(region.centroid[2])], yt[int(region.centroid[1])], zt[int(region.centroid[0])]),
                        points=points,  # Pass NumPy array
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
                        cloud_base_height=min_height  # This correctly assigns the lowest point's height
                    )
            
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
