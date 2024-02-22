from lib.cloud import Cloud
from skimage import measure
import numpy as np
from scipy.ndimage import binary_dilation

class CloudField:
    def __init__(self, l_data,  timestep, config):
        self.timestep = timestep
        labeled_array, self.clouds = self.identify_clouds(l_data, config)
        #print (f"original clouds: {self.clouds}")
        updated_labeled_array = self.merge_boundary_clouds(labeled_array, config)
        self.clouds = self.update_clouds_from_labeled_array(updated_labeled_array, l_data, config)
        #print (f"updated clouds: {self.clouds}")


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
        print ("Merging boundary clouds...")
        edge_clouds = self.identify_edge_clouds(labeled_array)
        merges = self.find_boundary_merges(edge_clouds, labeled_array, labeled_array.shape)
        updated_array = self.update_labels_for_merges(labeled_array, merges)
        return updated_array

    def identify_edge_clouds(self, labeled_array):
        print ("Identifying edge clouds...")
        edge_clouds = {
            'top': set(labeled_array[0, :].flatten()) - {0},
            'bottom': set(labeled_array[-1, :].flatten()) - {0},
            'left': set(labeled_array[:, 0].flatten()) - {0},
            'right': set(labeled_array[:, -1].flatten()) - {0}
        }
        print (f"Edge clouds at each boundary. top: {edge_clouds['top']}, bottom: {edge_clouds['bottom']}, left: {edge_clouds['left']}, right: {edge_clouds['right']}")
        return edge_clouds

    def find_boundary_merges(self, edge_clouds, labeled_array, array_shape):
        print ("Finding merges between edge clouds...")
        top_edge, bottom_edge, left_edge, right_edge = edge_clouds
        merges = []
        distance_threshold = 3

        # find merges between top and bottom edges
        for label in edge_clouds['top']:
            top_cloud_points = np.argwhere(labeled_array[0,:] == label)
            for bottom_label in edge_clouds['bottom']:
                bottom_cloud_points = np.argwhere(labeled_array[-1,:] == bottom_label)
                # check if any top cloud points are close to any bottom cloud points
                for top_point in top_cloud_points:
                    for bottom_point in bottom_cloud_points:
                        if abs(top_point[1] - bottom_point[1]) < distance_threshold:
                            merges.append((label, bottom_label))

        # find merges between left and right edges
        for label in edge_clouds['left']:
            left_cloud_points = np.argwhere(labeled_array[:,0] == label)
            for right_label in edge_clouds['right']:
                right_cloud_points = np.argwhere(labeled_array[:,-1] == right_label)
                # check if any left cloud points are close to any right cloud points
                for left_point in left_cloud_points:
                    for right_point in right_cloud_points:
                        if abs(left_point[0] - right_point[0]) < distance_threshold:
                            merges.append((label, right_label))

        return merges


    def update_labels_for_merges(self, labeled_array, merges):
        print ("Updating labels for merges...")
        for merge_pair in merges:
            # merge_pair is a tuple of two labels to be merged
            min_label = min(merge_pair)
            for label in merge_pair:
                labeled_array[labeled_array == label] = min_label
        return labeled_array


    def update_clouds_from_labeled_array(self, labeled_array, l_data, config):
        print ("Updating cloud data from labeled array...")
        # clear the current cloud data
        self.clouds.clear()

        # recalculate the clouds from the updated labeled array
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
        print(f"Cloud data for {len(clouds)} updated objects.")

        return clouds

