import utils.plotting_utils as plotting_utils
import math

class CloudTracker:
    def __init__(self, config):
        self.cloud_tracks = {}
        self.config = config

    def drift_translation_calculation(self):
        dx = int(self.config['u_drift'] * self.config['timestep_duration'])
        dy = int(self.config['v_drift'] * self.config['timestep_duration'])
        return dx, dy


    def update_tracks(self, current_cloud_field):
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

        # Create a set of expected points for last_cloud considering drift
        expected_last_cloud_points = {(x + dx, y + dy, z) for x, y, z in last_cloud_in_track.points}

        # Get points of the current cloud
        current_cloud_points = set(cloud.points)

        threshold = self.config['horizontal_resolution']
        threshold_squared = threshold ** 2

        for cx, cy, cz in current_cloud_points:
            for ex, ey, ez in expected_last_cloud_points:
                distance_squared = (ex - cx) ** 2 + (ey - cy) ** 2 + (ez - cz) ** 2
                if distance_squared <= threshold_squared:
                    return True
        return False

        # Visualize the points
        # last_cloud_points = set(last_cloud_in_track.points)
        #plotting_utils.visualize_points(last_cloud_points, expected_last_cloud_points, current_cloud_points)
        #plotting_utils.visualize_points_plotly(last_cloud_points,expected_last_cloud_points,current_cloud_points)


    def get_tracks(self):
        return self.cloud_tracks

