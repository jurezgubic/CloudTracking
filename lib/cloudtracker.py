import utils.plotting_utils as plotting_utils
import math
import numpy as np
import gc
from memory_profiler import profile
from scipy.spatial import cKDTree

class CloudTracker:
    """"Class to track clouds over time."""
    def __init__(self, config):
        self.cloud_tracks = {}
        self.config = config
        self.mean_u = None
        self.mean_v = None
        self.mean_w = None  # Add mean vertical velocity
        self.zt = None

    def drift_translation_calculation(self):
        """ Calculate drift translation based on the namelist drift values. """
        if self.config['switch_background_drift'] == True:
            dx = int(self.config['u_drift'] * self.config['timestep_duration']) # Drift in x-direction
            dy = int(self.config['v_drift'] * self.config['timestep_duration']) # Drift in y-direction
        else:
            dx = dy = 0
        return dx, dy


    def wind_drift_calculation(self, cz):
        """ Calculate wind drift based on the height of the cloud point using pre-loaded mean wind data. """
        if self.config['switch_wind_drift'] == True:
            z_index = np.argmin(np.abs(self.zt - cz))  # Find nearest z-level index
            wind_dx = self.mean_u[z_index] * self.config['timestep_duration'] # Wind drift in x-direction
            wind_dy = self.mean_v[z_index] * self.config['timestep_duration'] # Wind drift in y-direction
        else:
            wind_dx = wind_dy = 0
        return wind_dx, wind_dy

    def vertical_drift_calculation(self, cz):
        """Calculate vertical drift based on the height of the cloud point using pre-loaded mean vertical velocity data."""
        if self.config.get('switch_vertical_drift', True):  # Default to True if not specified
            z_index = np.argmin(np.abs(self.zt - cz))  # Find nearest z-level index
            vert_dz = self.mean_w[z_index] * self.config['timestep_duration']  # Vertical drift
        else:
            vert_dz = 0
        return vert_dz

    def update_tracks(self, current_cloud_field, mean_u, mean_v, mean_w, zt, xt, yt):
        """Update the cloud tracks with the current cloud field."""
        self.mean_u = mean_u
        self.mean_v = mean_v
        self.mean_w = mean_w
        self.zt = zt
        self.xt = xt
        self.yt = yt
        
        # Calculate domain dimensions (for use in boundary handling)
        self.domain_size_x = (self.xt[-1] - self.xt[0]) + self.config['horizontal_resolution']
        self.domain_size_y = (self.yt[-1] - self.yt[0]) + self.config['horizontal_resolution']
        
        new_matched_clouds = set()
        
        # Dictionary to track cloud inheritance - maps current cloud_id to list of (parent_cloud, parent_track_id)
        cloud_inheritance = {}

        if not self.cloud_tracks:  # First timestep
            for cloud_id, cloud in current_cloud_field.clouds.items():
                cloud.age = 0
                self.cloud_tracks[cloud_id] = [cloud]
        else:
            # FIRST PASS: Find all potential matches between previous and current clouds
            for track_id, track in list(self.cloud_tracks.items()):
                last_cloud_in_track = track[-1]
                if not last_cloud_in_track.is_active:
                    continue
                    
                # Find all potential fragments that match this cloud
                for cloud_id, cloud in current_cloud_field.clouds.items():
                    if self.is_match(cloud, last_cloud_in_track):
                        # Record this potential inheritance
                        if cloud_id not in cloud_inheritance:
                            cloud_inheritance[cloud_id] = []
                        cloud_inheritance[cloud_id].append((last_cloud_in_track, track_id))
            
            # SECOND PASS: Handle merges and splits with proper age inheritance
            # Process merges first (clouds with multiple potential parents)
            merge_candidates = {cid: parents for cid, parents in cloud_inheritance.items() if len(parents) > 1}
            
            for cloud_id, parent_list in merge_candidates.items():
                cloud = current_cloud_field.clouds[cloud_id]
                
                # Sort parents by age to find oldest
                parent_list.sort(key=lambda x: x[0].age, reverse=True)
                oldest_parent, oldest_parent_track_id = parent_list[0]
                
                # Continue oldest parent's track
                cloud.age = oldest_parent.age + 1
                self.cloud_tracks[oldest_parent_track_id].append(cloud)
                new_matched_clouds.add(cloud_id)
                
                # Mark other parents as merged
                for parent, parent_track_id in parent_list[1:]:
                    if parent_track_id != oldest_parent_track_id:
                        parent.is_active = False
                        parent.merged_into = oldest_parent_track_id
            
            # THIRD PASS: Process regular matches and splits
            for track_id, track in list(self.cloud_tracks.items()):
                last_cloud_in_track = track[-1]
                if not last_cloud_in_track.is_active:
                    continue
                    
                found_match = False
                
                # Find primary match to continue the track
                for cloud_id, cloud in current_cloud_field.clouds.items():
                    if cloud_id not in new_matched_clouds and cloud_id in cloud_inheritance:
                        # Check if this track is in the potential parents
                        for parent, parent_track_id in cloud_inheritance[cloud_id]:
                            if parent_track_id == track_id:
                                # Update max height if needed
                                current_max_height = max(z for _, _, z in cloud.points)
                                if current_max_height > last_cloud_in_track.max_height:
                                    last_cloud_in_track.max_height = current_max_height
                                
                                # Continue the track with this fragment
                                cloud.age = last_cloud_in_track.age + 1
                                track.append(cloud)
                                new_matched_clouds.add(cloud_id)
                                found_match = True
                                break
                        
                        if found_match:
                            break
                
                if not found_match:
                    # Mark as inactive if no matches found (and not already marked as merged)
                    if last_cloud_in_track.merged_into is None:
                        last_cloud_in_track.is_active = False
            
            # Handle remaining fragments (splits and new clouds)
            for cloud_id, cloud in current_cloud_field.clouds.items():
                if cloud_id not in new_matched_clouds:
                    if cloud_id in cloud_inheritance:
                        # This is a split cloud - inherit age from parent
                        # For consistency, use the oldest parent if multiple
                        parents = cloud_inheritance[cloud_id]
                        parents.sort(key=lambda x: x[0].age, reverse=True)
                        parent_cloud = parents[0][0]
                        cloud.age = parent_cloud.age + 1
                    else:
                        # This is a genuinely new cloud
                        cloud.age = 0
                    
                    # Start a new track
                    self.cloud_tracks[cloud_id] = [cloud]

    def match_clouds(self, current_cloud_field):
        """ Match clouds from the current cloud field to the existing tracks. """
        matched_clouds = set()
        for track_id, track in self.cloud_tracks.items(): # Check each existing track for a match in the current cloud field
            last_cloud_in_track = track[-1] # Get the last cloud in the track
            for cloud_id, cloud in current_cloud_field.clouds.items(): # Check if the cloud is a match
                if cloud_id not in matched_clouds and self.is_match(cloud, last_cloud_in_track): # If the cloud is a match
                    self.cloud_tracks[track_id].append(cloud) # Add the cloud to the track
                    matched_clouds.add(cloud_id) # Mark the cloud as matched
                    break
            else:
                # If no match is found, consider the cloud has dissipated or is out of bounds
                continue

        # Add new clouds as new tracks
        for cloud_id, cloud in current_cloud_field.clouds.items():
            if cloud_id not in matched_clouds:
                self.cloud_tracks[cloud_id] = [cloud]

    # @profile
    def is_match(self, cloud, last_cloud_in_track):
        """Check if the cloud is a match using dynamic thresholds based on maximum velocities."""
        # First, check if the last cloud is still active
        if not last_cloud_in_track.is_active:
            return False
        
        # Basic validation
        # Check if points are None or empty NumPy arrays
        if last_cloud_in_track.points is None or last_cloud_in_track.points.size == 0 or \
           cloud.points is None or cloud.points.size == 0:
            return False
            
        # Calculate horizontal drift
        dx, dy = self.drift_translation_calculation()
        
        # Calculate dynamic thresholds based on maximum velocities
        timestep_duration = self.config['timestep_duration']
        safety_factor = 2.0  # Buffer for turbulence and numerical effects
        
        # Find maximum velocities (get absolute max values)
        if self.mean_u is not None and self.mean_v is not None and self.mean_w is not None:
            max_u = np.max(np.abs(self.mean_u)) * timestep_duration * safety_factor
            max_v = np.max(np.abs(self.mean_v)) * timestep_duration * safety_factor
            max_w = np.max(np.abs(self.mean_w)) * timestep_duration * safety_factor
            
            # Set minimum thresholds in case velocities are very small
            horizontal_threshold = max(max(max_u, max_v), self.config['horizontal_resolution'])
            vertical_threshold = max_w*2  # Use vertical drift as a threshold
        else:
            # Fallback to config values if mean velocities aren't available
            horizontal_threshold = self.config['horizontal_resolution']
            vertical_threshold = self.config['horizontal_resolution'] * 1
        
        # Vectorized approach for iterating through points by height
        previous_cloud_points_np = last_cloud_in_track.points
        # Basic validation for previous_cloud_points_np should have already caught empty arrays,
        # but a check here can be a safeguard if called directly or logic changes.
        if previous_cloud_points_np.size == 0:
            return False 

        unique_z_coords_previous = np.unique(previous_cloud_points_np[:, 2])

        # Track if we find any boundary crossing match
        found_boundary_crossing = False # This is for logging/debugging boundary crossings

        # Create a KD-tree for each height level of the previous cloud after applying drift
        for z_prev_level in unique_z_coords_previous:
            points_at_prev_z = previous_cloud_points_np[previous_cloud_points_np[:, 2] == z_prev_level]
            
            if points_at_prev_z.shape[0] == 0: # Should not happen if unique_z_coords_previous is from points
                continue

            # Calculate wind drift for this height
            wind_dx, wind_dy = self.wind_drift_calculation(z_prev_level)
            vert_dz = self.vertical_drift_calculation(z_prev_level)
            
            adjusted_dx = dx + wind_dx
            adjusted_dy = dy + wind_dy
            adjusted_dz = vert_dz
            
            # Apply drift to points in 3D (vectorized)
            drift_vector = np.array([adjusted_dx, adjusted_dy, adjusted_dz])
            adjusted_points_at_prev_z = points_at_prev_z + drift_vector
            
            # Build KD-tree with 3D points
            if adjusted_points_at_prev_z.shape[0] == 0:
                continue
            tree = cKDTree(adjusted_points_at_prev_z)
            
            # Define vertical search range for current cloud points based on the dynamic vertical threshold
            z_min_search = z_prev_level - vertical_threshold
            z_max_search = z_prev_level + vertical_threshold + adjusted_dz # Account for vertical drift
            
            # Get 3D query points from current cloud near this height range (vectorized)
            current_cloud_points_np = cloud.points
            if current_cloud_points_np.size == 0: # Should be caught by initial validation
                query_points_in_range = np.array([])
            else:
                mask_z_range = (current_cloud_points_np[:, 2] >= z_min_search) & (current_cloud_points_np[:, 2] <= z_max_search)
                query_points_in_range = current_cloud_points_np[mask_z_range]
            
            # Cyclic boundary handling
            if query_points_in_range.shape[0] > 0:
                # Create extended query points with domain boundary wrapping
                # extended_query_points = list(query_points_in_range) # OLD
                
                # # Wrap points near domain boundaries # OLD
                # for pt in query_points_in_range: # OLD
                #     x_qp, y_qp, cz_qp = pt # OLD
                # ... (rest of the old manual wrapping)

                # NEW Vectorized approach for extending query points:
                points_to_extend_list = [query_points_in_range]
                x_coords_qp, y_coords_qp, z_coords_qp = query_points_in_range[:, 0], query_points_in_range[:, 1], query_points_in_range[:, 2]

                # X-wrapping
                wrap_x_right_edge_mask = x_coords_qp < self.xt[0] + horizontal_threshold # Points near left edge, wrap to right
                wrap_x_left_edge_mask = x_coords_qp > self.xt[-1] - horizontal_threshold # Points near right edge, wrap to left
                
                if np.any(wrap_x_right_edge_mask):
                    points_to_extend_list.append(np.column_stack((x_coords_qp[wrap_x_right_edge_mask] + self.domain_size_x, y_coords_qp[wrap_x_right_edge_mask], z_coords_qp[wrap_x_right_edge_mask])))
                if np.any(wrap_x_left_edge_mask):
                    points_to_extend_list.append(np.column_stack((x_coords_qp[wrap_x_left_edge_mask] - self.domain_size_x, y_coords_qp[wrap_x_left_edge_mask], z_coords_qp[wrap_x_left_edge_mask])))

                # Y-wrapping
                wrap_y_bottom_edge_mask = y_coords_qp < self.yt[0] + horizontal_threshold # Points near top edge, wrap to bottom
                wrap_y_top_edge_mask = y_coords_qp > self.yt[-1] - horizontal_threshold # Points near bottom edge, wrap to top

                if np.any(wrap_y_bottom_edge_mask):
                    points_to_extend_list.append(np.column_stack((x_coords_qp[wrap_y_bottom_edge_mask], y_coords_qp[wrap_y_bottom_edge_mask] + self.domain_size_y, z_coords_qp[wrap_y_bottom_edge_mask])))
                if np.any(wrap_y_top_edge_mask):
                    points_to_extend_list.append(np.column_stack((x_coords_qp[wrap_y_top_edge_mask], y_coords_qp[wrap_y_top_edge_mask] - self.domain_size_y, z_coords_qp[wrap_y_top_edge_mask])))
                
                # Corner wrapping (XY)
                # Near top-left (xt[0], yt[0]), wrap to (+domain_size_x, +domain_size_y)
                wrap_tl_mask = wrap_x_right_edge_mask & wrap_y_bottom_edge_mask
                if np.any(wrap_tl_mask):
                    points_to_extend_list.append(np.column_stack((x_coords_qp[wrap_tl_mask] + self.domain_size_x, y_coords_qp[wrap_tl_mask] + self.domain_size_y, z_coords_qp[wrap_tl_mask])))
                # Near bottom-left (xt[0], yt[-1]), wrap to (+domain_size_x, -domain_size_y)
                wrap_bl_mask = wrap_x_right_edge_mask & wrap_y_top_edge_mask
                if np.any(wrap_bl_mask):
                    points_to_extend_list.append(np.column_stack((x_coords_qp[wrap_bl_mask] + self.domain_size_x, y_coords_qp[wrap_bl_mask] - self.domain_size_y, z_coords_qp[wrap_bl_mask])))
                # Near top-right (xt[-1], yt[0]), wrap to (-domain_size_x, +domain_size_y)
                wrap_tr_mask = wrap_x_left_edge_mask & wrap_y_bottom_edge_mask
                if np.any(wrap_tr_mask):
                    points_to_extend_list.append(np.column_stack((x_coords_qp[wrap_tr_mask] - self.domain_size_x, y_coords_qp[wrap_tr_mask] + self.domain_size_y, z_coords_qp[wrap_tr_mask])))
                # Near bottom-right (xt[-1], yt[-1]), wrap to (-domain_size_x, -domain_size_y)
                wrap_br_mask = wrap_x_left_edge_mask & wrap_y_top_edge_mask
                if np.any(wrap_br_mask):
                    points_to_extend_list.append(np.column_stack((x_coords_qp[wrap_br_mask] - self.domain_size_x, y_coords_qp[wrap_br_mask] - self.domain_size_y, z_coords_qp[wrap_br_mask])))
                
                extended_query_points = np.vstack(points_to_extend_list) if len(points_to_extend_list) > 1 else query_points_in_range

                # Check if standard matching would work (no boundary crossing)
                # Query with original points in range to see if a non-boundary match exists
                distances_orig, _ = tree.query(query_points_in_range, k=1, distance_upper_bound=horizontal_threshold + 1e-9) # Add epsilon for strict inequality
                orig_match = np.any(distances_orig <= horizontal_threshold)
                
                # Check if extended matching would work (includes boundary crossing)
                # Query with all extended points
                distances_extended, _ = tree.query(extended_query_points, k=1, distance_upper_bound=horizontal_threshold + 1e-9)
                extended_match = np.any(distances_extended <= horizontal_threshold)
                
                # Detect pure boundary crossing (matches only with wrapped points)
                if extended_match and not orig_match:
                    found_boundary_crossing = True # Set flag for potential logging
                    print(f"BOUNDARY CROSSING DETECTED! Track: {last_cloud_in_track.cloud_id}, Cloud: {cloud.cloud_id}")
                    print(f"  Horizontal threshold: {horizontal_threshold}")
                    print(f"  Domain size: {self.domain_size_x} x {self.domain_size_y}")
                    print(f"  Cloud points near boundary: {len(extended_query_points) - len(query_points_in_range)}")
                    
                    # Additional debug information
                    if hasattr(cloud, 'location') and hasattr(last_cloud_in_track, 'location'):
                        print(f"  Current cloud location: {cloud.location}")
                        print(f"  Previous cloud location: {last_cloud_in_track.location}")
                        
                        # Calculate expected location with drift
                        expected_x = last_cloud_in_track.location[0] + adjusted_dx
                        expected_y = last_cloud_in_track.location[1] + adjusted_dy
                        print(f"  Expected location with drift: ({expected_x}, {expected_y})")
                        
                        # Check which boundary was crossed
                        if abs(cloud.location[0] - expected_x) > self.domain_size_x/2:
                            print(f"  X-boundary crossed (E-W)")
                        if abs(cloud.location[1] - expected_y) > self.domain_size_y/2:
                            print(f"  Y-boundary crossed (N-S)")
                
                # Return True if we find any match (boundary crossing or normal)
                if extended_match:
                    return True # Match found for this z-level, so clouds match
    
        return False # No match found across all z-levels

        # Visualize the points for background drift trabnslation (belongs to is_match function)
        # expected_last_cloud_points = {(x + dx, y + dy, z) for x, y, z in last_cloud_in_track.points}
        # current_cloud_points = set(cloud.points)
        # last_cloud_points = set(last_cloud_in_track.points)
        #plotting_utils.visualize_points(last_cloud_points, expected_last_cloud_points, current_cloud_points)
        #plotting_utils.visualize_points_plotly(last_cloud_points,expected_last_cloud_points,current_cloud_points)


    def get_tracks(self):
        return self.cloud_tracks

