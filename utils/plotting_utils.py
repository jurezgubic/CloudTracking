import matplotlib.pyplot as plt
from netCDF4 import Dataset

def plot_cloud_sizes(filename):
    with Dataset(filename, 'r') as dataset:
        sizes = dataset.variables['size'][:]
        plt.plot(sizes)
        plt.xlabel('Cloud Index')
        plt.ylabel('Size')
        plt.title('Cloud Sizes')
        plt.show()

# Additional plotting functions can be added here

