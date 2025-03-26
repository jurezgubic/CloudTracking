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

    def update_tracks(self, current_cloud_field, mean_u, mean_v, mean_w, zt):
        """Update the cloud tracks with the current cloud field."""
        self.mean_u = mean_u
        self.mean_v = mean_v
        self.mean_w = mean_w
        self.zt = zt
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
        if not cloud.points or not last_cloud_in_track.points:
            return False
            
        # Calculate horizontal drift
        dx, dy = self.drift_translation_calculation()
        
        # Calculate dynamic thresholds based on maximum velocities
        timestep_duration = self.config['timestep_duration']
        safety_factor = 1.5  # Buffer for turbulence and numerical effects
        
        # Find maximum velocities (get absolute max values)
        if self.mean_u is not None and self.mean_v is not None and self.mean_w is not None:
            max_u = np.max(np.abs(self.mean_u)) * timestep_duration * safety_factor
            max_v = np.max(np.abs(self.mean_v)) * timestep_duration * safety_factor
            max_w = np.max(np.abs(self.mean_w)) * timestep_duration * safety_factor
            
            # Set minimum thresholds in case velocities are very small
            horizontal_threshold = max(max(max_u, max_v), self.config['horizontal_resolution'])
            vertical_threshold = max(max_w, self.config['horizontal_resolution'] * 2)
        else:
            # Fallback to config values if mean velocities aren't available
            horizontal_threshold = self.config['horizontal_resolution']
            vertical_threshold = self.config['horizontal_resolution'] * 3
        
        # Group points by height for wind drift calculation
        height_to_points = {}
        for x, y, z in last_cloud_in_track.points:
            if z not in height_to_points:
                height_to_points[z] = []
            height_to_points[z].append((x, y, z))
        
        # Create a KD-tree for each height level with 3D drift
        for z, points in height_to_points.items():
            # Calculate wind drift for this height
            wind_dx, wind_dy = self.wind_drift_calculation(z)
            vert_dz = self.vertical_drift_calculation(z)  # Calculate vertical drift
            
            adjusted_dx = dx + wind_dx
            adjusted_dy = dy + wind_dy
            adjusted_dz = vert_dz  # Vertical displacement
            
            # Apply drift to points in 3D
            adjusted_points = np.array([(x + adjusted_dx, y + adjusted_dy, z + adjusted_dz) for x, y, z in points])
            
            # Build KD-tree with 3D points
            tree = cKDTree(adjusted_points)
            
            # Define vertical search range based on the dynamic vertical threshold
            z_min = z - vertical_threshold
            z_max = z + vertical_threshold + adjusted_dz
            
            # Get 3D query points from current cloud near this height
            query_points = np.array([(x, y, cz) for x, y, cz in cloud.points 
                                   if z_min <= cz <= z_max])
            
            if len(query_points) > 0:
                # Find if any points are within threshold distance
                distances, _ = tree.query(query_points, k=1)
                if np.any(distances <= horizontal_threshold):
                    return True
                    
        return False

        # Visualize the points for background drift trabnslation (belongs to is_match function)
        # expected_last_cloud_points = {(x + dx, y + dy, z) for x, y, z in last_cloud_in_track.points}
        # current_cloud_points = set(cloud.points)
        # last_cloud_points = set(last_cloud_in_track.points)
        #plotting_utils.visualize_points(last_cloud_points, expected_last_cloud_points, current_cloud_points)
        #plotting_utils.visualize_points_plotly(last_cloud_points,expected_last_cloud_points,current_cloud_points)


    def get_tracks(self):
        return self.cloud_tracks

