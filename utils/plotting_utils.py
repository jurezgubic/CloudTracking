import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np

def plot_cloud_sizes(filename):
    with Dataset(filename, 'r') as dataset:
        sizes = dataset.variables['size'][:]
        plt.plot(sizes)
        plt.xlabel('Cloud Index')
        plt.ylabel('Size')
        plt.title('Cloud Sizes')
        plt.show()


def plot_labeled_regions(stage_of_labeling, labeled_array, timestep, plot_all_levels=False, specific_level=0):
    """
    Plot labeled regions from a 3D labeled array for a specific timestep,
    and save the plot as a PNG file.

    Parameters:
    - labeled_array: 3D numpy array with labeled regions.
    - timestep: Integer or string, specifies the timestep for the plot's filename.
    - plot_all_levels: Boolean, if True, plots all z-levels, otherwise plots a specific level.
    - specific_level: Integer, specifies which z-level to plot if plot_all_levels is False.
    """
    if plot_all_levels:
        # Aggregate all levels by taking the maximum label at each (x, y) across all z-levels
        max_label_projection = np.amax(labeled_array, axis=0)
        plot_title = f"Aggregate of All Levels at Timestep {timestep}"
        filename = f"{stage_of_labeling}_timestep_{timestep}_aggregate.png"
        _plot_single_level(max_label_projection, title=plot_title, filename=filename)
    else:
        # Plot a specific z-level
        plot_title = f"Name {stage_of_labeling} Level {specific_level} at Timestep {timestep}"
        filename = f"{stage_of_labeling}_timestep_{timestep}_level_{specific_level}.png"
        _plot_single_level(labeled_array[specific_level, :, :], title=plot_title, filename=filename)

def _plot_single_level(slice_array, title="", filename="plot.png"):
    """
    Helper function to plot a single 2D slice of labeled regions and save it as a PNG file.

    Parameters:
    - slice_array: 2D numpy array with labeled regions for a specific z-level or aggregate.
    - title: String, title for the plot.
    - filename: String, filename for saving the plot.
    """
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(slice_array)[1:]  # Exclude background (0)

    for label in unique_labels:
        # Find points for the current label
        y, x = np.where(slice_array == label)
        plt.scatter(x, y, alpha=0.6, edgecolors='w', s=5)  # Adjust marker size with `s`

        # Calculate the centroid of the cloud points for annotation
        centroid_x = np.mean(x)
        centroid_y = np.mean(y)
        plt.text(centroid_x, centroid_y, str(label), color='black', fontsize=9, ha='center', va='center')

    # Horizontal line at max and min y
    plt.axhline(slice_array.shape[0], color='gray', linestyle='--')
    plt.axvline(slice_array.shape[1], color='gray', linestyle='--')
    # Vertical line at max and min x
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(title)
    # Omitting the legend as individual labels are annotated on the plot
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the plot figure



