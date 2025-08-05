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
        self.zt = None
        self.xt = None
        self.yt = None
        self.domain_size_x = None
        self.domain_size_y = None
        self.tainted_tracks = set()  # Store IDs of tracks with incomplete lifecycles
        self.track_id_to_index = {}  # Maps track_id to stable NetCDF index
        self.next_index = 0          # Next available NetCDF index

    def update_tracks(self, current_cloud_field, zt, xt, yt):
        """Update the cloud tracks with the current cloud field."""
        self.zt = zt
        self.xt = xt
        self.yt = yt
        
        # Calculate domain dimensions (for use in boundary handling)
        if self.domain_size_x is None:
            self.domain_size_x = (self.xt[-1] - self.xt[0]) + self.config['horizontal_resolution']
            self.domain_size_y = (self.yt[-1] - self.yt[0]) + self.config['horizontal_resolution']
        
        new_matched_clouds = set()
        
        # Dictionary to track cloud inheritance
        cloud_inheritance = {}

        if not self.cloud_tracks:  # First timestep
            for cloud_id, cloud in current_cloud_field.clouds.items():
                cloud.age = 0
                self.cloud_tracks[cloud_id] = [cloud]
        else:
            # Log whether pre-filtering is enabled
            if self.config.get('use_pre_filtering', True):
                print("Using centroid pre-filtering for cloud matching")
            else:
                print("Pre-filtering disabled - checking all possible matches")
                
            # Pre-filter potential matches considering periodic boundaries
            potential_matches = self.pre_filter_cloud_matches(current_cloud_field)
            
            # FIRST PASS: Process pre-filtered matches
            if self.config.get('use_batch_processing', True):
                # --- Use batch processing for better performance ---
                import time
                batch_size = self.config.get('batch_size', 50)
                batch_start = time.time()
                print(f"Using batch processing with batch size {batch_size}")
                
                # Process matches in batches
                batch_results = self.batch_process_matches(potential_matches, current_cloud_field, batch_size)
                
                # Convert batch results to cloud_inheritance format (exact same structure as original code)
                for (cloud_id, track_id), is_match in batch_results.items():
                    if is_match:
                        cloud = current_cloud_field.clouds[cloud_id]
                        last_cloud_in_track = self.cloud_tracks[track_id][-1]
                        
                        if cloud_id not in cloud_inheritance:
                            cloud_inheritance[cloud_id] = []
                        cloud_inheritance[cloud_id].append((last_cloud_in_track, track_id))
                
                batch_end = time.time()
                print(f"Batch processing completed in {batch_end - batch_start:.2f} seconds")
            else:
                # --- Original sequential processing logic ---
                for track_id, candidate_cloud_ids in potential_matches.items():
                    last_cloud_in_track = self.cloud_tracks[track_id][-1]
                    
                    # Only check pre-filtered candidate clouds
                    for cloud_id in candidate_cloud_ids:
                        cloud = current_cloud_field.clouds[cloud_id]
                        if self.is_match(cloud, last_cloud_in_track, current_cloud_field):
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
                                current_max_height = np.max(cloud.points[:, 2])
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

        # Add after all matching is complete
        new_clouds_count = 0
        matched_clouds_count = 0
        merged_clouds_count = 0
        split_clouds_count = 0
        inactive_clouds_count = 0

        for track_id, track in self.cloud_tracks.items():
            last_cloud = track[-1]
            if not last_cloud.is_active:
                inactive_clouds_count += 1
            elif last_cloud.timestep == current_cloud_field.timestep:
                if len(track) == 1:
                    new_clouds_count += 1
                elif hasattr(last_cloud, 'is_split') and last_cloud.is_split:
                    split_clouds_count += 1
                elif hasattr(last_cloud, 'is_merged') and last_cloud.is_merged:
                    merged_clouds_count += 1
                else:
                    matched_clouds_count += 1

        print(f"Cloud tracking summary:")
        print(f"  New clouds: {new_clouds_count}")
        print(f"  Matched clouds: {matched_clouds_count}")
        print(f"  Merged clouds: {merged_clouds_count}")
        print(f"  Split clouds: {split_clouds_count}")
        print(f"  Inactive clouds: {inactive_clouds_count}")

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
    def is_match(self, cloud, last_cloud_in_track, current_cloud_field):
        """
        Checks if a cloud in the current timestep matches a cloud from the previous timestep
        by searching for overlapping SURFACE points within a cylindrical volume.
        
        Args:
            cloud: Cloud object from current timestep
            last_cloud_in_track: Cloud object from previous timestep
            current_cloud_field: CloudField object containing the pre-built KD-tree
        """
        # --- 1. Validation and Threshold Calculation ---
        if not last_cloud_in_track.is_active or not last_cloud_in_track.surface_points.any():
            return False
        
        if current_cloud_field.surface_points_kdtree is None:
            return False

        timestep_duration = self.config['timestep_duration']
        
        # The safety factor creates a buffer around the predicted location.
        # It accounts for the cloud's acceleration/deceleration between timesteps.
        safety_factor = self.config.get('match_safety_factor', 2.0) 

        # --- Calculate a dynamic search radius based on the cloud's OWN velocity. ---
        # This ensures the search area is proportional to the cloud's specific momentum.
        # A cloud with high velocity will have a larger search radius.
        u_abs = abs(last_cloud_in_track.mean_u)
        v_abs = abs(last_cloud_in_track.mean_v)
        w_abs = abs(last_cloud_in_track.mean_w)

        horizontal_threshold = max(u_abs, v_abs) * timestep_duration * safety_factor
        vertical_threshold = w_abs * timestep_duration * safety_factor
        
        # Ensure the threshold is at least one grid cell to handle stationary clouds.
        horizontal_threshold = max(horizontal_threshold, 2 * self.config['horizontal_resolution'])
        vertical_threshold = max(vertical_threshold, 2 * self.config['horizontal_resolution'])

        # --- 2. Prepare Point Sets and Apply Drift (Vectorized) ---
        last_points = last_cloud_in_track.surface_points
        
        # --- Apply drift to previous cloud points ---
        dx = last_cloud_in_track.mean_u * timestep_duration
        dy = last_cloud_in_track.mean_v * timestep_duration
        dz = last_cloud_in_track.mean_w * timestep_duration
        adjusted_points = last_points + [dx, dy, dz]
        
        # --- 3. Handle Cyclic Boundaries (Vectorized) ---
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
        
        # --- 4. Query the pre-built global KD-tree ---
        nearby_indices_list = current_cloud_field.surface_points_kdtree.query_ball_point(
            all_query_points[:, :2], r=horizontal_threshold
        )
        
        # --- 5. Check for matches with the current cloud ID ---
        cloud_id = cloud.cloud_id
        for i, indices in enumerate(nearby_indices_list):
            if not indices:
                continue
                
            # Get cloud IDs for the nearby points
            nearby_cloud_ids = np.unique(current_cloud_field.surface_point_to_cloud_id[indices])
            
            # If the current cloud's ID is among them, check vertical proximity
            if cloud_id in nearby_cloud_ids:
                # Get only points belonging to the current cloud
                current_cloud_mask = current_cloud_field.surface_point_to_cloud_id[indices] == cloud_id
                current_point_indices = np.array(indices)[current_cloud_mask]
                
                if len(current_point_indices) == 0:
                    continue
                    
                # Check vertical proximity
                last_z = all_query_points[i, 2]
                current_z_values = current_cloud_field.surface_points_array[current_point_indices, 2]
                
                if np.any(np.abs(current_z_values - last_z) <= vertical_threshold):
                    return True
    
        return False

    def pre_filter_cloud_matches(self, current_cloud_field):
        """
        Pre-filter potential cloud matches based on centroid proximity to reduce
        the number of expensive is_match() calls, while properly handling periodic boundaries.
        
        When pre-filtering is disabled, returns all possible combinations.
        
        Returns a dictionary mapping track_ids to lists of candidate cloud_ids.
        """
        # Check if pre-filtering is disabled in config
        if not self.config.get('use_pre_filtering', True):
            # Return all possible combinations when disabled
            potential_matches = {}
            for track_id, track in self.cloud_tracks.items():
                last_cloud = track[-1]
                if last_cloud.is_active:
                    # All current clouds are potential matches
                    potential_matches[track_id] = list(current_cloud_field.clouds.keys())
            return potential_matches
        
        # Original pre-filtering logic continues below
        potential_matches = {}
        
        # Skip if no tracks or clouds exist
        if not self.cloud_tracks or not current_cloud_field.clouds:
            return potential_matches
        
        # Create arrays of cloud centroids
        prev_centroids = []
        prev_track_ids = []
        
        # Get centroids of active clouds from previous timestep
        for track_id, track in self.cloud_tracks.items():
            last_cloud = track[-1]
            if last_cloud.is_active:
                prev_centroids.append(last_cloud.location)
                prev_track_ids.append(track_id)
        
        # Skip if no active clouds
        if not prev_centroids:
            return potential_matches
            
        prev_centroids = np.array(prev_centroids)
        
        # Get current cloud centroids
        curr_centroids = []
        curr_cloud_ids = []
        
        for cloud_id, cloud in current_cloud_field.clouds.items():
            curr_centroids.append(cloud.location)
            curr_cloud_ids.append(cloud_id)
        
        # Skip if no current clouds
        if not curr_centroids:
            return potential_matches
            
        curr_centroids = np.array(curr_centroids)
        
        # Physics-based search radius calculation from config
        timestep_duration = self.config['timestep_duration']
        max_speed = self.config.get('max_expected_cloud_speed')
        
        # Apply safety factor for the search radius from config
        safety_factor = self.config.get('bounding_box_safety_factor')
        search_radius = max_speed * timestep_duration * safety_factor
        
        # For each previous cloud, find potential matches
        for i, prev_centroid in enumerate(prev_centroids):
            track_id = prev_track_ids[i]
            candidate_ids = []
            
            # Extract coordinates for better readability
            px, py, pz = prev_centroid
            
            # Check each current cloud, considering periodic boundaries
            for j, (curr_centroid, cloud_id) in enumerate(zip(curr_centroids, curr_cloud_ids)):
                cx, cy, cz = curr_centroid
                
                # Handle x-direction periodic boundary
                dx = abs(cx - px)
                if dx > self.domain_size_x / 2:
                    dx = self.domain_size_x - dx
                    
                # Handle y-direction periodic boundary
                dy = abs(cy - py)
                if dy > self.domain_size_y / 2:
                    dy = self.domain_size_y - dy
                    
                # Direct distance in z (non-periodic)
                dz = abs(cz - pz)
                
                # Calculate effective horizontal and vertical distances
                horiz_dist = np.sqrt(dx*dx + dy*dy)
                vert_dist = dz
                
                # Check if within search thresholds
                if horiz_dist <= search_radius and vert_dist <= search_radius:
                    candidate_ids.append(cloud_id)
            
            if candidate_ids:
                potential_matches[track_id] = candidate_ids

        total_candidates = sum(len(candidates) for candidates in potential_matches.values())
        print(f"Pre-filtering found {total_candidates} potential matches across {len(potential_matches)} active tracks")
    
        return potential_matches

    def get_tracks(self):
        return self.cloud_tracks

    def batch_process_matches(self, potential_matches, current_cloud_field, batch_size=50):
        """
        Process potential matches in configurable-sized batches to balance memory usage and performance.
        
        Args:
            potential_matches: Dictionary mapping track_ids to lists of candidate cloud_ids
            current_cloud_field: CloudField object containing the KD-tree
            batch_size: Number of (track_id, cloud_id) pairs to process in each batch
            
        Returns:
            Dictionary mapping (cloud_id, track_id) to True/False indicating match status
        """
        import time  # Add at top of file if not already there
        
        # Skip if KD-tree doesn't exist
        if current_cloud_field.surface_points_kdtree is None:
            return {}
        
        # Create flattened list of all (track_id, cloud_id) pairs to process
        all_pairs = []
        for track_id, cloud_ids in potential_matches.items():
            for cloud_id in cloud_ids:
                all_pairs.append((track_id, cloud_id))
        
        total_pairs = len(all_pairs)
        print(f"Processing {total_pairs} potential matches in batches of {batch_size}")
        
        # Process in batches
        batch_results = {}
        
        for batch_start in range(0, total_pairs, batch_size):
            batch_end = min(batch_start + batch_size, total_pairs)
            current_batch = all_pairs[batch_start:batch_end]
            
            batch_start_time = time.time()
            print(f"Processing batch {batch_start//batch_size + 1} of {(total_pairs + batch_size - 1)//batch_size}: " 
                  f"pairs {batch_start+1}-{batch_end} of {total_pairs}")
            
            # Process this batch
            batch_matches = self._process_match_batch(current_batch, current_cloud_field)
            
            # Update overall results
            batch_results.update(batch_matches)
            
            batch_end_time = time.time()
            print(f"  Batch completed in {batch_end_time - batch_start_time:.2f} seconds, " 
                  f"found {len(batch_matches)} matches")
            
            # Optional: Force garbage collection after each batch
            # import gc
            # gc.collect()
        
        return batch_results

    def _process_match_batch(self, batch_pairs, current_cloud_field):
        """
        Process a single batch of potential matches.
        
        Args:
            batch_pairs: List of (track_id, cloud_id) pairs to check in this batch
            current_cloud_field: CloudField object containing the KD-tree
            
        Returns:
            Dictionary of match results for this batch
        """
        batch_results = {}
        timestep_duration = self.config['timestep_duration']
        safety_factor = self.config.get('match_safety_factor', 2.0)
        
        # Prepare data structures for batch processing
        all_query_points = []
        query_metadata = []  # (cloud_id, track_id, original_point_idx)
        thresholds = []
        
        # For each pair in the batch
        for track_id, cloud_id in batch_pairs:
            # Get the relevant objects
            last_cloud_in_track = self.cloud_tracks[track_id][-1]
            cloud = current_cloud_field.clouds[cloud_id]
            
            # Skip invalid combinations early
            if (not last_cloud_in_track.is_active or 
                not last_cloud_in_track.surface_points.any()):
                continue
            
            # Calculate cloud-specific thresholds
            u_abs = abs(last_cloud_in_track.mean_u)
            v_abs = abs(last_cloud_in_track.mean_v)
            w_abs = abs(last_cloud_in_track.mean_w)
            
            horizontal_threshold = max(u_abs, v_abs) * timestep_duration * safety_factor
            vertical_threshold = w_abs * timestep_duration * safety_factor
            
            # Minimum threshold for stationary clouds
            horizontal_threshold = max(horizontal_threshold, 2 * self.config['horizontal_resolution'])
            vertical_threshold = max(vertical_threshold, 2 * self.config['horizontal_resolution'])
            
            # Apply drift to previous cloud points
            last_points = last_cloud_in_track.surface_points
            dx = last_cloud_in_track.mean_u * timestep_duration
            dy = last_cloud_in_track.mean_v * timestep_duration
            dz = last_cloud_in_track.mean_w * timestep_duration
            adjusted_points = last_points + [dx, dy, dz]
            
            # Handle cyclic boundaries
            points_sets = [adjusted_points]
            if np.any(adjusted_points[:, 0] < self.xt[0] + horizontal_threshold):
                points_sets.append(adjusted_points + [self.domain_size_x, 0, 0])
            if np.any(adjusted_points[:, 0] > self.xt[-1] - horizontal_threshold):
                points_sets.append(adjusted_points - [self.domain_size_x, 0, 0])
            if np.any(adjusted_points[:, 1] < self.yt[0] + horizontal_threshold):
                points_sets.append(adjusted_points + [0, self.domain_size_y, 0])
            if np.any(adjusted_points[:, 1] > self.yt[-1] - horizontal_threshold):
                points_sets.append(adjusted_points - [0, self.domain_size_y, 0])
            
            # Add all points to the query batch
            for point_set in points_sets:
                for i, point in enumerate(point_set):
                    all_query_points.append(point)
                    query_metadata.append((cloud_id, track_id, i))
                    thresholds.append((horizontal_threshold, vertical_threshold))
        
        # Skip if no valid query points
        if not all_query_points:
            return batch_results
        
        # Convert to numpy arrays for vectorized processing
        all_query_points = np.array(all_query_points)
        horizontal_thresholds = np.array([t[0] for t in thresholds])
        
        # Execute batch KD-tree query
        nearby_indices_list = current_cloud_field.surface_points_kdtree.query_ball_point(
            all_query_points[:, :2], 
            r=horizontal_thresholds
        )
        
        # Process query results
        for i, indices in enumerate(nearby_indices_list):
            if not indices:
                continue
                
            cloud_id, track_id, point_idx = query_metadata[i]
            horizontal_threshold, vertical_threshold = thresholds[i]
            
            # If match already found for this pair, skip
            if (cloud_id, track_id) in batch_results:
                continue
            
            # Get cloud IDs for nearby points
            nearby_cloud_ids = np.unique(current_cloud_field.surface_point_to_cloud_id[indices])
            
            # Check if the cloud is among the nearby points
            if cloud_id in nearby_cloud_ids:
                # Get only points belonging to the current cloud
                current_cloud_mask = current_cloud_field.surface_point_to_cloud_id[indices] == cloud_id
                current_point_indices = np.array(indices)[current_cloud_mask]
                
                if len(current_point_indices) == 0:
                    continue
                
                # Check vertical proximity
                last_z = all_query_points[i, 2]
                current_z_values = current_cloud_field.surface_points_array[current_point_indices, 2]
                
                if np.any(np.abs(current_z_values - last_z) <= vertical_threshold):
                    batch_results[(cloud_id, track_id)] = True
        
        return batch_results

