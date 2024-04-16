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
        dx = int(self.config['u_drift'] * self.config['timestep_duration'])
        dy = int(self.config['v_drift'] * self.config['timestep_duration'])
        return dx, dy


    def wind_drift_calculation(self, cz):
        """ Calculate wind drift based on the z-coordinate of the cloud point using pre-loaded mean wind data. """
        z_index = np.argmin(np.abs(self.zt - cz))  # Find nearest z-level index
        wind_dx = int(self.mean_u[z_index] * self.config['timestep_duration'])
        wind_dy = int(self.mean_v[z_index] * self.config['timestep_duration'])
        return wind_dx, wind_dy

    def update_tracks(self, current_cloud_field, mean_u, mean_v, zt):
        self.mean_u = mean_u
        self.mean_v = mean_v
        self.zt = zt
        if not self.cloud_tracks:  # If this is the first timestep
            for cloud_id, cloud in current_cloud_field.clouds.items():
                self.cloud_tracks[cloud_id] = [cloud]
        else:
            self.match_clouds(current_cloud_field)

    def match_clouds(self, current_cloud_field):
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
        dx, dy = self.drift_translation_calculation() # Drift calculation per timestep

        threshold = self.config['horizontal_resolution']
        threshold_squared = threshold ** 2

        dx, dy = self.drift_translation_calculation()
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

