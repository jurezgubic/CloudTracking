from lib.cloud import Cloud
from skimage import measure
import numpy as np
from scipy.ndimage import binary_dilation

class CloudField:
    def __init__(self, l_data,  timestep, config):
        self.timestep = timestep
        # creates a labeled array of objects
        labeled_array = self.identify_regions(l_data, config)
        # identify edge regions
        edge_regions = self.identify_edge_regions(labeled_array)
        # find merges between edge regions
        merges = self.find_boundary_merges(edge_regions, labeled_array, labeled_array.shape)
        # update labels for merges
        updated_labeled_array = self.update_labels_for_merges(labeled_array, merges)
        # create cloud data from updated labeled array
        self.clouds = self.create_clouds_from_labeled_array(updated_labeled_array, l_data, config)


    def identify_regions(self, l_data, config):
        condition = l_data > config['l_condition']
        labeled_array = measure.label(condition, connectivity=3)
        num_features = np.max(labeled_array)
        print(f"Objects labeled. Number of initial features: {num_features}")

        return labeled_array

    def identify_edge_regions(self, labeled_array):
        edge_regions = {
            'top': set(labeled_array[0, :].flatten()) - {0},
            'bottom': set(labeled_array[-1, :].flatten()) - {0},
            'left': set(labeled_array[:, 0].flatten()) - {0},
            'right': set(labeled_array[:, -1].flatten()) - {0}
        }
        print (f"Edge regions: top: {edge_regions['top']}, bottom: {edge_regions['bottom']}, left: {edge_regions['left']}, right: {edge_regions['right']}")
        return edge_regions





    def find_boundary_merges(self, edge_regions, labeled_array, array_shape):
        merges = []
        distance_threshold = 3

        # Extract points for top and bottom boundaries only if there are labels present
        top_region_labels = edge_regions['top']
        bottom_region_labels = edge_regions['bottom']

        # Prepare arrays for top and bottom boundary points if labels are present, else use empty arrays
        top_points = np.vstack([np.argwhere(labeled_array[0, :] == label) \
                                for label in top_region_labels]) \
                                if top_region_labels else np.empty((0, 2))
        bottom_points = np.vstack([np.argwhere(labeled_array[-1, :] == label) \
                                for label in bottom_region_labels]) \
                                if bottom_region_labels else np.empty((0, 2))


        # Check for empty arrays to avoid errors in the next steps
        if top_points.size > 0 and bottom_points.size > 0:

            # Adjust points to consider only the x-coordinate for comparison
            top_x = top_points[:, 1][:, np.newaxis]  # Reshape for broadcasting
            bottom_x = bottom_points[:, 1]

            # Vectorized comparison: calculate the absolute difference between x-coordinates
            diff_matrix = np.abs(top_x - bottom_x)

            # Identify matches: where any difference is within the threshold
            matches = np.any(diff_matrix <= self.distance_threshold, axis=1)

            # Extract labels for matched top boundary points
            matched_top_labels = np.unique(top_points[matches, 0])

            # For each matched top label, find corresponding bottom labels
            for top_label in matched_top_labels:
                # Find bottom labels with matching points within threshold
                matched_bottom_labels = np.unique(bottom_points[np.any(diff_matrix[matches] \
                                        <= self.distance_threshold, axis=0), 0])

                # Append matched pairs to merges list
                for bottom_label in matched_bottom_labels:
                    merges.append((top_label, bottom_label))


        return merges

    def update_labels_for_merges(self, labeled_array, merges):
        print ("Updating labels for merges...")
        for merge_pair in merges:
            # merge_pair is a tuple of two labels to be merged
            min_label = min(merge_pair)
            for label in merge_pair:
                labeled_array[labeled_array == label] = min_label
        return labeled_array


    def create_clouds_from_labeled_array(self, updated_labeled_array, l_data, config):
        print ("Creating cloud data from labeled array...")

        # recalculate the clouds from the updated labeled array
        regions = measure.regionprops(updated_labeled_array, l_data)

        # Create Cloud objects for each identified cloud
        clouds = {}
        for region in regions:
            if region.area >= config['min_size']:
                cloud_id = f"{self.timestep}-{region.label}"
                cloud_mask = updated_labeled_array == region.label
                points = np.argwhere(cloud_mask)
                points = [tuple(point) for point in points]

                # Estimate the surface area
                surface_mask = binary_dilation(cloud_mask) & ~cloud_mask
                surface_area = np.sum(surface_mask)

                clouds[cloud_id] = Cloud(
                    cloud_id=cloud_id,
                    size=region.area,
                    surface_area=surface_area,
                    location=(region.centroid[2], region.centroid[1], region.centroid[0]),
                    points=points,
                    timestep=self.timestep
                )
        print(f"Cloud data for {len(clouds)} objects.")

        return clouds

