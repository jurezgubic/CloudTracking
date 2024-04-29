from skimage import measure
import numpy as np
import gc
from memory_profiler import profile
from scipy.ndimage import binary_dilation
from utils.plotting_utils import plot_labeled_regions
from lib.cloud import Cloud


class CloudField:
    """Class to identify and track clouds in a labeled data field."""
    def __init__(self, l_data, w_data,  timestep, config, xt, yt, zt):
        self.timestep = timestep
        self.distance_threshold = config['distance_threshold']
        self.xt = xt
        self.yt = yt
        self.zt = zt

        # creates a labeled array of objects
        labeled_array = self.identify_regions(l_data, w_data, config)

        if config['plot_switch'] == True:
            plot_labeled_regions('labeled', labeled_array, timestep=timestep, plot_all_levels=True)

        # find merges between edge regions
        merges = self.find_boundary_merges(labeled_array)

        # update labels for merges
        updated_labeled_array = self.update_labels_for_merges(labeled_array, merges)

        # create cloud data from updated labeled array
        self.clouds = self.create_clouds_from_labeled_array(
            updated_labeled_array, l_data, w_data, config, xt, yt, zt)

        # plot the updated labeled clouds if plot_switch is True
        if config['plot_switch'] == True:
            plot_labeled_regions('updated', updated_labeled_array, timestep=timestep, plot_all_levels=True)

        # example of how to plot labeled regions
        # plot_labeled_regions(
            #'name_or_array', labeled_array, timestep=timestep, plot_all_levels=False, specific_level=50)
        # plot_labeled_regions(
            #'name_of_array', labeled_array, timestep=timestep, plot_all_levels=True) #plots all levels



    def identify_regions(self, l_data, w_data, config):
        """Identify cloudy regions in the labeled data."""
        if config['w_switch'] == True:
            condition = (l_data > config['l_condition']) & (w_data >= config['w_condition'])
        else:
            condition = l_data > config['l_condition']

        # Label the regions
        labeled_array = measure.label(condition, connectivity=3)
        num_features = np.max(labeled_array)
        print(f"Objects labeled. Number of initial features: {num_features}")

        return labeled_array



    def find_boundary_merges(self, labeled_array):
        """Find merges between regions on opposite boundaries of the labeled array."""
        merges = []
        z, y, x = labeled_array.shape
        threshold = self.distance_threshold

        # Function to find boundary points for a given label
        def boundary_points(label, boundary_array):
            return np.argwhere(boundary_array == label)

        # Function to check if points in one array are within threshold of points in another
        def check_within_threshold(points_a, points_b, threshold):
            for point_a in points_a:
                # Broadcasting difference calculation over points_b
                diff = np.abs(points_b - point_a)
                # Check if any point_b is within threshold distance of point_a
                if np.any(np.all(diff <= threshold, axis=1)):
                    return True
            return False

        # Preparing boundary arrays
        north_boundary = labeled_array[:, -1, :]
        south_boundary = labeled_array[:, 0, :]
        east_boundary = labeled_array[:, :, -1]
        west_boundary = labeled_array[:, :, 0]

        # Identifying unique labels on each boundary
        north_labels = np.unique(north_boundary)[1:]  # Excluding background
        south_labels = np.unique(south_boundary)[1:]
        east_labels = np.unique(east_boundary)[1:]
        west_labels = np.unique(west_boundary)[1:]

        # print all booundary labels
        print(f"North labels: {north_labels}")
        print(f"South labels: {south_labels}")
        print(f"East labels: {east_labels}")
        print(f"West labels: {west_labels}")

        # Distance threshold for considering two points as "close"
        threshold = np.array([self.distance_threshold, self.distance_threshold])

        # North-South Merges
        for n_label in north_labels:
            n_points = boundary_points(n_label, north_boundary)
            for s_label in south_labels:
                s_points = boundary_points(s_label, south_boundary)
                if check_within_threshold(n_points, s_points, threshold):
                    merges.append((n_label, s_label))

        # East-West Merges
        for e_label in east_labels:
            e_points = boundary_points(e_label, east_boundary)
            for w_label in west_labels:
                w_points = boundary_points(w_label, west_boundary)
                if check_within_threshold(e_points, w_points, threshold):
                    merges.append((e_label, w_label))


        print(f"Merges: {merges}")
        return merges



    def update_labels_for_merges(self, labeled_array, merges):
        """Update labels for merging regions."""
        print ("Updating labels for merges...")
        for merge_pair in merges:
            # merge_pair is a tuple of two labels to be merged
            min_label = min(merge_pair)
            for label in merge_pair:
                labeled_array[labeled_array == label] = min_label
        return labeled_array

    # @profile
    def create_clouds_from_labeled_array(self, updated_labeled_array, l_data, w_data, config, xt, yt, zt):
        """Create Cloud objects from the updated labeled array."""
        print ("Creating cloud data from labeled array...")

        # recalculate the clouds from the updated labeled array
        regions = measure.regionprops(updated_labeled_array, l_data)

        #find height index of cloud base
        base_index = np.abs(zt - config['cloud_base_altitude']).argmin()

        # Create Cloud objects for each identified cloud
        clouds = {} # dictionary to store cloud objects
        for region in regions: # iterate over all regions
            if region.area >= config['min_size']: # only consider regions larger than min_size
                # print (f"Processing cloud {region.label}...")

                cloud_id = f"{self.timestep}-{region.label}" # unique id for the cloud
                cloud_mask = updated_labeled_array == region.label # mask for the current region
                point_indices = np.argwhere(cloud_mask) # indices of points in the region
                points = [(xt[x], yt[y], zt[z]) for z, y, x in point_indices] # coordinates of points in the region

                # extract values of w at the points
                w_values = [w_data[z, y, x] for z, y, x in point_indices]
                # Calculate vertical velocity variables
                max_w = np.max(w_values)
                base_w_values = [w_data[z, y, x] for z, y, x in point_indices if zt[z] == zt[base_index]] # vertical velocity values at cloud base
                max_w_cloud_base = np.max(base_w_values) if base_w_values else np.nan # max vertical velocity at cloud base

                # estimate area of cloud base
                base_points = [point for point in points if point[2] == zt[base_index]] # points at cloud base
                cloud_base_area = len(base_points)*config['horizontal_resolution']**2 # area of cloud base

                # estimate max height of cloud
                max_height = np.max([point[2] for point in points])

                # Estimate the surface area of the cloud
                surface_mask = binary_dilation(cloud_mask) & ~cloud_mask
                surface_area = np.sum(surface_mask)

                # calculate mass flux
                mass_flux = sum(w_data[z, y, x] * l_data[z, y, x] for z, y, x in point_indices)

                # Create a Cloud object and store it in the dictionary
                clouds[cloud_id] = Cloud(
                    cloud_id=cloud_id,
                    size=region.area,
                    surface_area=surface_area,
                    location=(region.centroid[2], region.centroid[1], region.centroid[0]),
                    points=points,
                    max_height=max_height,
                    max_w=max_w,
                    max_w_cloud_base=max_w_cloud_base,
                    cloud_base_area=cloud_base_area,
                    mass_flux = mass_flux,
                    timestep=self.timestep
                )
        print(f"Cloud data for {len(clouds)} objects.")

        # print cloud data for debugging purposes if print_cloud_data_switch is True
        print_cloud_data_switch = False
        if print_cloud_data_switch == True:
            # print some of the data that the object cloud has
            for cloud in clouds:
                # print a short line to separate the clouds
                print("-------------------------------------------------------")
                print(f"cloud id is {clouds[cloud].cloud_id}")
                print(f"size is {clouds[cloud].size}")
                print(f"surface area is {clouds[cloud].surface_area}")
                print(f"location is {clouds[cloud].location}")
                print(f"timestep is {clouds[cloud].timestep}")
                print(clouds[cloud].points)

        return clouds
