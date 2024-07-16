import utils.plotting_utils as plotting_utils
import math
import numpy as np
import gc
from memory_profiler import profile


class CloudTracker:
    """"Class to track clouds over time."""
    def __init__(self, config):
        self.cloud_tracks = {}
        self.config = config
        self.mean_u = None
        self.mean_v = None
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


    def update_tracks(self, current_cloud_field, mean_u, mean_v, zt):
        """ Update the cloud tracks with the current cloud field. """
        self.mean_u = mean_u
        self.mean_v = mean_v
        self.zt = zt
        new_matched_clouds = set()

        if not self.cloud_tracks:  # If this is the first timestep
            for cloud_id, cloud in current_cloud_field.clouds.items(): # Add all clouds as new tracks
                cloud.age = 0 # initialise cloud age
                self.cloud_tracks[cloud_id] = [cloud] # Add the cloud as a new track
        else:
            # check each existing track for a match in the current cloud field
            for track_id, track in list(self.cloud_tracks.items()):
                last_cloud_in_track = track[-1]
                found_match = False

                for cloud_id, cloud in current_cloud_field.clouds.items(): # Check if the cloud is a match
                    if cloud_id not in new_matched_clouds and self.is_match(cloud, last_cloud_in_track): # If the cloud is a match
                        # print ("Processing cloud ID: ", cloud_id)
                        current_max_height = max(z for _, _, z in cloud.points) # Update the max height of the cloud
                        if current_max_height > last_cloud_in_track.max_height: # If the current cloud is higher
                            last_cloud_in_track.max_height = current_max_height # Update the max height of the cloud
                        cloud.age = last_cloud_in_track.age + 1 # increment the age of the cloud
                        track.append(cloud) # Add the cloud to the track
                        new_matched_clouds.add(cloud_id) # Mark the cloud as matched
                        found_match = True
                        break

                if not found_match:
                    # ensure that only the currenty instance of cloud is marked as inactive
                    last_cloud_in_track.is_active = False  # Mark the last cloud as inactive

            # Add new clouds as new tracks
            for cloud_id, cloud in current_cloud_field.clouds.items(): # Add all unmatched clouds as new tracks
                if cloud_id not in new_matched_clouds: # If the cloud is not matched
                    cloud.age = 0 # initialise cloud age to 0
                    self.cloud_tracks[cloud_id] = [cloud] # Add the cloud as a new track



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
        """ Check if the cloud is a match to the last cloud in the track.
        This is done by checking if any point in the current cloud is within the
        horizontal resolution of the last cloud in the track. """

        dx, dy = self.drift_translation_calculation() # Drift calculation per timestep

        # Calculate the threshold for horizontal resolution
        threshold = self.config['horizontal_resolution']
        threshold_squared = threshold ** 2 # Squared for faster calculation

        # Check if any point in the current cloud is within the threshold of the last cloud in the track
        for cx, cy, cz in cloud.points:
            wind_dx, wind_dy = self.wind_drift_calculation(cz) # Wind drift calculation per timestep
            # Check if the point is within the threshold of the last cloud in the track
            for ex, ey, ez in last_cloud_in_track.points:
                adjusted_dx = dx + wind_dx
                adjusted_dy = dy + wind_dy
                # Check if the point is within the threshold of the last cloud in the track
                if ((ex - cx + adjusted_dx) ** 2 + (ey - cy + adjusted_dy) ** 2) <= threshold_squared:
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

