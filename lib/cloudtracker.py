class CloudTracker:
    def __init__(self):
        self.cloud_tracks = {}

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

    # I wonder if it might be best to translate the cloud field here by u.v from the namelist

    def is_match(self, cloud, last_cloud_in_track):
        # Calculate overlap between cloud points
        current_cloud_points = set(cloud.points)
        last_cloud_points = set(last_cloud_in_track.points)
        overlap = current_cloud_points.intersection(last_cloud_points)

        # Consider it a match if there is any overlap
        return len(overlap) > 0

    def get_tracks(self):
        return self.cloud_tracks

