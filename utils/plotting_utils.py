import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from datetime import datetime


def visualize_points(last_cloud_points, expected_last_cloud_points, current_cloud_points):
    """used to visualise:
        cloud field from the timestep n+1 (current cloud)
        cloud field of timestep n (last cloud points) and
        the shifted cloud of timestep n (expected points post drift).
        Primary use in cloudtracker.py in is_match functions.
        Produces an interactive 3D plot using matplotlib - mid simulation!
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot last cloud points
    if last_cloud_points is not None and last_cloud_points.size > 0:
        lx, ly, lz = last_cloud_points[:, 0], last_cloud_points[:, 1], last_cloud_points[:, 2]
        ax.scatter(lx, ly, lz, color='r', label='Last Cloud Points')

    # Plot expected positions after drift
    if expected_last_cloud_points is not None and expected_last_cloud_points.size > 0:
        ex, ey, ez = expected_last_cloud_points[:, 0], expected_last_cloud_points[:, 1], expected_last_cloud_points[:, 2]
        ax.scatter(ex, ey, ez, color='g', alpha=0.5, label='Expected Points Post-Drift')

    # Plot current cloud points
    if current_cloud_points is not None and current_cloud_points.size > 0:
        cx, cy, cz = current_cloud_points[:, 0], current_cloud_points[:, 1], current_cloud_points[:, 2]
        ax.scatter(cx, cy, cz, color='b', alpha=0.5, label='Current Cloud Points')

    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('Cloud Point Comparison')
    plt.show()


def visualize_points_plotly(last_cloud_points, expected_last_cloud_points, current_cloud_points):
    """used to visualise:
        cloud field from the timestep n+1 (current cloud)
        cloud field of timestep n (last cloud points) and
        the shifted cloud of timestep n (expected points post drift).
        Primary use in cloudtracker.py in is_match functions.
        Produces interactive 3D plots and saves them using current datetime.
    """
    fig = go.Figure()

    # Add last cloud points
    if last_cloud_points is not None and last_cloud_points.size > 0:
        lx, ly, lz = last_cloud_points[:, 0], last_cloud_points[:, 1], last_cloud_points[:, 2]
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz, mode='markers', marker=dict(size=4, color='red'), 
            name='Last Cloud Points'))

    # Add expected points after drift
    if expected_last_cloud_points is not None and expected_last_cloud_points.size > 0:
        ex, ey, ez = expected_last_cloud_points[:, 0], expected_last_cloud_points[:, 1], expected_last_cloud_points[:, 2]
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode='markers', marker=dict(size=4, color='green', opacity=0.5),
            name='Expected Points Post-Drift'))

    # Add current cloud points
    if current_cloud_points is not None and current_cloud_points.size > 0:
        cx, cy, cz = current_cloud_points[:, 0], current_cloud_points[:, 1], current_cloud_points[:, 2]
        fig.add_trace(go.Scatter3d(
            x=cx, y=cy, z=cz, mode='markers', marker=dict(size=4, color='blue', opacity=0.5), 
            name='Current Cloud Points'))

    fig.update_layout(title='Cloud Point Comparison', margin=dict(l=0, r=0, b=0, t=30))
    fig.write_html('cloud_points_plot.html')  # Save the figure as an HTML file
     # Generate a unique filename based on the current datetime
    filename = f"cloud_points_plot_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.html"
    fig.write_html(filename)  # Save the figure as an HTML file



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



