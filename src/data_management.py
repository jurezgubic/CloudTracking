from netCDF4 import Dataset
import numpy as np
from lib.cloudfield import CloudField


def load_cloud_field_from_file(l_file_path, timestep, config):
    """
    Load cloud data from files for a specific timestep and create a CloudField object.

    Returns:
    - A CloudField object for the given timestep.
    """
    # Load 'l' data
    l_dataset = Dataset(l_file_path, 'r')
    l_data = l_dataset.variables['l'][timestep, :, :, :]

    # Correctly create and return the CloudField object
    return CloudField(l_data, timestep, config)

