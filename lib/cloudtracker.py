import utils.plotting_utils as plotting_utils
import math
import numpy as np

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
            dx = int(self.config['u_drift'] * self.config['timestep_duration'])
            dy = int(self.config['v_drift'] * self.config['timestep_duration'])
        else:
            dx = dy = 0
        return dx, dy


    def wind_drift_calculation(self, cz):
        """ Calculate wind drift based on the height of the cloud point using pre-loaded mean wind data. """
        if self.config['switch_wind_drift'] == True:
            z_index = np.argmin(np.abs(self.zt - cz))  # Find nearest z-level index
            wind_dx = self.mean_u[z_index] * self.config['timestep_duration']
            wind_dy = self.mean_v[z_index] * self.config['timestep_duration']
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
            for cloud_id, cloud in current_cloud_field.clouds.items():
                self.cloud_tracks[cloud_id] = [cloud]
        else:
            for track_id, track in list(self.cloud_tracks.items()):
                last_cloud_in_track = track[-1]
                found_match = False

                for cloud_id, cloud in current_cloud_field.clouds.items():
                    if cloud_id not in new_matched_clouds and self.is_match(cloud, last_cloud_in_track):
                        track.append(cloud)
                        new_matched_clouds.add(cloud_id)
                        found_match = True
                        break

                if not found_match:
                    last_cloud_in_track.is_active = False  # Mark the last cloud as inactive

            # Add new clouds as new tracks
            for cloud_id, cloud in current_cloud_field.clouds.items():
                if cloud_id not in new_matched_clouds:
                    self.cloud_tracks[cloud_id] = [cloud]

        #if not self.cloud_tracks:  # If this is the first timestep
        #    for cloud_id, cloud in current_cloud_field.clouds.items():
        #        self.cloud_tracks[cloud_id] = [cloud]
        #else:
        #    self.match_clouds(current_cloud_field)


    def match_clouds(self, current_cloud_field):
        """ Match clouds from the current cloud field to the existing tracks. """
        matched_clouds = set()
        for track_id, track in self.cloud_tracks.items():
            last_cloud_in_track = track[-1]
            for cloud_id, cloud in current_cloud_field.clouds.items():
                if cloud_id not in matched_clouds and self.is_match(cloud, last_cloud_in_track):
                    self.cloud_tracks[track_id].append(cloud)
                    matched_clouds.add(cloud_id)
                    break
            else:
                # If no match is found, consider the cloud has dissipated or is out of bounds
                continue

        # Add new clouds as new tracks
        for cloud_id, cloud in current_cloud_field.clouds.items():
            if cloud_id not in matched_clouds:
                self.cloud_tracks[cloud_id] = [cloud]


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
            wind_dx, wind_dy = self.wind_drift_calculation(cz)
            for ex, ey, ez in last_cloud_in_track.points:
                adjusted_dx = dx + wind_dx
                adjusted_dy = dy + wind_dy
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

