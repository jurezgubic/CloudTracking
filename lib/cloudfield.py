from lib.cloud import Cloud
from skimage import measure
import numpy as np
from scipy.ndimage import binary_dilation

class CloudField:
    def __init__(self, l_data,  timestep, config):
        self.timestep = timestep
        labeled_array, self.clouds = self.identify_clouds(l_data, config)
        self.labeled_array = self.merge_boundary_clouds(labeled_array, config)
        self.update_clouds_from_labeled_array()


    def identify_clouds(self, l_data, config):
        condition = l_data > config['l_condition']

        # Label the objects in the array
        labeled_array = measure.label(condition, connectivity=3)
        num_features = np.max(labeled_array)
        print(f"Objects labeled. Number of initial features: {num_features}")

        # Calculate properties of the labeled regions
        regions = measure.regionprops(labeled_array, l_data)

        # Create Cloud objects for each identified cloud
        clouds = {}
        for region in regions:
            if region.area >= config['min_size']:
                cloud_id = f"{self.timestep}-{region.label}"
                cloud_mask = labeled_array == region.label
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

        return labeled_array, clouds

    def merge_boundary_clouds(self, labeled_array, config):
        edge_clouds = self.identify_edge_clouds(labeled_array)
        merges = self.find_boundary_merges(edge_clouds, labeled_array.shape)
        updated_array = self.update_labels_for_merges(labeled_array, merges)
        return labeled_array

    def identify_edge_clouds(self, labeled_array):
        top_edge = set(labeled_array[0, :].flatten())
        bottom_edge = set(labeled_array[-1, :].flatten())
        left_edge = set(labeled_array[:, 0].flatten())
        right_edge = set(labeled_array[:, -1].flatten())
        return top_edge, bottom_edge, left_edge, right_edge

    def find_boundary_merges(self, edge_clouds, array_shape):
        # ---todo---. needs work! This is just a placeholder
        # this merges anything with anything, add something that checks location
        top_edge, bottom_edge, left_edge, right_edge = edge_clouds
        merges = []
        if top_edge & bottom_edge:
            merges.append((top_edge & bottom_edge, 'y merge'))
        if left_edge & right_edge:
            merges.append((left_edge & right_edge, 'x merge'))
        return merges


    def update_labels_for_merges(self, labeled_array, merges):
        for merge_set, direction in merges:
            for label in merge_set:
                # ---todo--- also needs changing
                labeled_array[labeled_array == label] = min(merge_set)
        return labeled_array

    def update_clouds_from_labeled_array(self):
        # ---todo---. needs work! This is just a placeholder
        # should update self.clouds to reflect the merges. 
        # This might involve recalculating cloud properties and ensuring cloud IDs are consistent.
        pass
