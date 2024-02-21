from lib.cloud import Cloud
from skimage import measure
import numpy as np
from scipy.ndimage import binary_dilation

class CloudField:
    def __init__(self, l_data,  timestep, config):
        self.timestep = timestep
        self.clouds = self.identify_clouds(l_data, config)

    def identify_clouds(self, l_data, config):
        print("Starting cloud identification...")

        # Create a condition array based on the threshold
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
        return clouds

