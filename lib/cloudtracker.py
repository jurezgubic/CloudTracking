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

    def wind_drift_calculation(self, cz_array):
        """ Calculate wind drift based on the height of the cloud point using pre-loaded mean wind data. """
        if self.config['switch_wind_drift'] and self.mean_u is not None and self.mean_v is not None:
            # Find nearest z-level indices for an array of z coordinates
            z_indices = np.argmin(np.abs(self.zt[:, np.newaxis] - cz_array), axis=0)
            wind_dx = self.mean_u[z_indices] * self.config['timestep_duration'] # Wind drift in x-direction
            wind_dy = self.mean_v[z_indices] * self.config['timestep_duration'] # Wind drift in y-direction
        else:
            wind_dx = wind_dy = np.zeros_like(cz_array, dtype=float)
        return wind_dx, wind_dy

    def vertical_drift_calculation(self, cz_array):
        """Calculate vertical drift based on the height of the cloud point using pre-loaded mean vertical velocity data."""
        if self.config.get('switch_vertical_drift', True) and self.mean_w is not None:  # Default to True if not specified
            # Find nearest z-level indices for an array of z coordinates
            z_indices = np.argmin(np.abs(self.zt[:, np.newaxis] - cz_array), axis=0)
            vert_dz = self.mean_w[z_indices] * self.config['timestep_duration']  # Vertical drift
        else:
            vert_dz = np.zeros_like(cz_array, dtype=float)
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

    # This function is deprecated by the more complex logic in update_tracks
    # but is kept here for reference or simpler tracking scenarios.
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
        """
        Checks if a cloud in the current timestep matches a cloud from the previous timestep
        by searching for overlapping SURFACE points within a cylindrical volume.
        """
        # --- 1. Validation and Threshold Calculation ---
        # Use surface_points for validation. .any() is needed for numpy arrays.
        if not last_cloud_in_track.is_active or not cloud.surface_points.any() or not last_cloud_in_track.surface_points.any():
            return False

        timestep_duration = self.config['timestep_duration']
        safety_factor = 2.0

        if self.mean_u is not None and self.mean_v is not None and self.mean_w is not None:
            max_u_abs = np.max(np.abs(self.mean_u))
            max_v_abs = np.max(np.abs(self.mean_v))
            max_w_abs = np.max(np.abs(self.mean_w))
            
            horizontal_threshold = max(max(max_u_abs, max_v_abs) * timestep_duration * safety_factor, self.config['horizontal_resolution'])
            vertical_threshold = max(max_w_abs * timestep_duration * safety_factor, self.config['horizontal_resolution'])
        else:
            horizontal_threshold = self.config['horizontal_resolution'] * 2
            vertical_threshold = self.config['horizontal_resolution']

        # --- 2. Prepare Point Sets and Apply Drift (Vectorized) ---
        # Use the much smaller set of surface_points for matching
        current_points = cloud.surface_points
        last_points = last_cloud_in_track.surface_points

        # --- Physics: Advect the previous cloud using its own internal mean velocity ---
        # This is a more accurate physical model than using the environmental mean,
        # as it accounts for the cloud's own momentum.
        dx = last_cloud_in_track.mean_u * timestep_duration
        dy = last_cloud_in_track.mean_v * timestep_duration
        dz = last_cloud_in_track.mean_w * timestep_duration
        
        # Apply total drift to get the expected position of the previous cloud's points
        # Note: np.copy is not needed here as addition creates a new array.
        adjusted_points = last_points + [dx, dy, dz]

        # --- 3. Build 2D KD-Tree for Horizontal Search ---
        # ToDo (optimisation): For better performance, build one KD-tree for the entire 
        # current_cloud_field and query it for each track, rather than building a tree for each potential match.
        tree_current_2d = cKDTree(current_points[:, :2]) # Use only X, Y coordinates

        # --- 4. Handle Cyclic Boundaries (Vectorized) ---
        points_to_query = [adjusted_points]
        # Check and append points for each boundary crossing scenario
        if np.any(adjusted_points[:, 0] < self.xt[0] + horizontal_threshold):
            points_to_query.append(adjusted_points + [self.domain_size_x, 0, 0])
        if np.any(adjusted_points[:, 0] > self.xt[-1] - horizontal_threshold):
            points_to_query.append(adjusted_points - [self.domain_size_x, 0, 0])
        if np.any(adjusted_points[:, 1] < self.yt[0] + horizontal_threshold):
            points_to_query.append(adjusted_points + [0, self.domain_size_y, 0])
        if np.any(adjusted_points[:, 1] > self.yt[-1] - horizontal_threshold):
            points_to_query.append(adjusted_points - [0, self.domain_size_y, 0])
        
        all_query_points = np.vstack(points_to_query)

        # --- 5. Perform Cylindrical Search ---
        # Find all pairs of points that are within the HORIZONTAL threshold using the 2D tree.
        nearby_indices_list = tree_current_2d.query_ball_point(all_query_points[:, :2], r=horizontal_threshold)

        # --- 6. Check Vertical Proximity for Horizontal Matches ---
        for i, current_point_indices in enumerate(nearby_indices_list):
            if not current_point_indices:  # No horizontal matches for this point.
                continue

            last_z = all_query_points[i, 2]
            current_z_values = current_points[current_point_indices, 2];

            # Check if the absolute vertical distance is within the threshold for ANY of the pairs.
            if np.any(np.abs(current_z_values - last_z) <= vertical_threshold):
                return True  # Found a valid match.

        return False # No match found.

    def get_tracks(self):
        return self.cloud_tracks

