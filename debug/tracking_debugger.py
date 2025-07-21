import sys
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting
import numpy as np

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_management import load_cloud_field_from_file
from lib.cloudtracker import CloudTracker

def visualise_match_attempt(cloud_t0, cloud_field_t1, tracker, config, xt, yt, zt):
    """
    Visualises the matching attempt by plotting the 3D points of the
    initial cloud and its potential successor with interactive controls.

    Args:
        cloud_t0: The Cloud object from the first timestep.
        cloud_field_t1: The CloudField object for the subsequent timestep.
        tracker: The CloudTracker instance containing settings.
        config: The main configuration dictionary.
        xt: X-coordinate array.
        yt: Y-coordinate array.
        zt: Z-coordinate array.
    """
    # Use larger figure size and add a message about interactivity
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add a text box explaining the navigation controls
    plt.figtext(0.5, 0.01, 
                "Interactive Controls: Left-click drag to rotate | Right-click drag to zoom | Middle-click drag to pan",
                ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # --- Convert cloud_t0 points to physical coordinates and plot ---
    points_t0 = cloud_t0.points
    
    # Debug information
    print(f"Points type: {type(points_t0)}")
    print(f"Points shape/length: {len(points_t0)}")
    
    if points_t0:
        print(f"First element type: {type(points_t0[0])}")
        print(f"First element value: {points_t0[0]}")
    
    # Extract x, y, z coordinates from the list of tuples
    if points_t0:
        # Points are already in physical coordinates as tuples (x, y, z)
        x_coords_t0 = [point[0] for point in points_t0]
        y_coords_t0 = [point[1] for point in points_t0]
        z_coords_t0 = [point[2] for point in points_t0]
        
        # Plot all cloud points
        ax.scatter(x_coords_t0, y_coords_t0, z_coords_t0, c='green', marker='o', s=30, alpha=0.5, label=f'Cloud {cloud_t0.cloud_id} (t)')
        
        # Also plot the cloud center as a larger marker
        loc = cloud_t0.location
        ax.scatter([loc[0]], [loc[1]], [loc[2]], c='darkgreen', marker='o', s=150, alpha=1.0, label=f'Center {cloud_t0.cloud_id} (t)')
    
    # --- Find the matched cloud and plot its points ---
    match_found = False
    matched_cloud_t1 = None
    for cloud_t1 in cloud_field_t1.clouds.values():
        if tracker.is_match(cloud_t1, cloud_t0):
            matched_cloud_t1 = cloud_t1
            match_found = True
            break
    
    if match_found:
        # Plot all points of the matched cloud
        points_t1 = matched_cloud_t1.points
        if points_t1:
            x_coords_t1 = [point[0] for point in points_t1]
            y_coords_t1 = [point[1] for point in points_t1]
            z_coords_t1 = [point[2] for point in points_t1]
            
            ax.scatter(x_coords_t1, y_coords_t1, z_coords_t1, c='yellow', marker='^', s=30, alpha=0.5, label=f'Cloud {matched_cloud_t1.cloud_id} (t+1)')
            
            # Also plot the cloud center as a larger marker
            loc_t1 = matched_cloud_t1.location
            ax.scatter([loc_t1[0]], [loc_t1[1]], [loc_t1[2]], c='orange', marker='^', s=150, alpha=1.0, label=f'Center {matched_cloud_t1.cloud_id} (t+1)')
            
            # Draw a line connecting the cloud centers
            ax.plot([loc[0], loc_t1[0]], [loc[1], loc_t1[1]], [loc[2], loc_t1[2]], 'k--', alpha=0.8, linewidth=2)
            
            # Calculate bounding box for both clouds to set view limits
            all_x = x_coords_t0 + x_coords_t1
            all_y = y_coords_t0 + y_coords_t1
            all_z = z_coords_t0 + z_coords_t1
            
            if all_x and all_y and all_z:
                padding = 500  # meters
                ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
                ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
                ax.set_zlim(min(all_z) - padding, max(all_z) + padding)
    else:
        # If no match, plot all nearby clouds within a reasonable distance
        loc = cloud_t0.location
        nearby_clouds = []
        
        for cloud_id, cloud_t1 in cloud_field_t1.clouds.items():
            loc_t1 = cloud_t1.location
            distance = np.sqrt((loc[0] - loc_t1[0])**2 + 
                              (loc[1] - loc_t1[1])**2 + 
                              (loc[2] - loc_t1[2])**2)
            
            # Threshold for "nearby" - adjust as needed
            if distance < 2000:  # meters
                nearby_clouds.append(cloud_t1)
                # Plot cloud center
                ax.scatter([loc_t1[0]], [loc_t1[1]], [loc_t1[2]], c='blue', marker='x', s=100, 
                          alpha=0.8, label=f'Candidate {cloud_id.split("-")[1]} (t+1)')
                
                # Add distance annotation
                ax.text(loc_t1[0], loc_t1[1], loc_t1[2] + 50, f"{distance:.0f}m", fontsize=8)
        
        # Plot the points of the closest candidate cloud
        if nearby_clouds:
            closest_cloud = min(nearby_clouds, key=lambda c: np.sqrt((loc[0] - c.location[0])**2 + 
                                                                    (loc[1] - c.location[1])**2 + 
                                                                    (loc[2] - c.location[2])**2))
            
            points_t1 = closest_cloud.points
            if points_t1:
                x_coords_t1 = [point[0] for point in points_t1]
                y_coords_t1 = [point[1] for point in points_t1]
                z_coords_t1 = [point[2] for point in points_t1]
                
                ax.scatter(x_coords_t1, y_coords_t1, z_coords_t1, c='blue', marker='^', s=30, 
                          alpha=0.3, label=f'Closest Candidate {closest_cloud.cloud_id.split("-")[1]}')
                
                # Calculate bounding box
                all_x = x_coords_t0 + x_coords_t1
                all_y = y_coords_t0 + y_coords_t1
                all_z = z_coords_t0 + z_coords_t1
                
                if all_x and all_y and all_z:
                    padding = 500  # meters
                    ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
                    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
                    ax.set_zlim(min(all_z) - padding, max(all_z) + padding)

    # If limits haven't been set by the bounding box calculations above
    if not hasattr(ax, '_xlim') or ax._xlim is None:
        # Set view limits around the current cloud
        padding = 1000  # meters
        loc = cloud_t0.location
        ax.set_xlim(loc[0] - padding, loc[0] + padding)
        ax.set_ylim(loc[1] - padding, loc[1] + padding)
        ax.set_zlim(loc[2] - padding, loc[2] + padding)

    # Set better 3D viewing angle
    ax.view_init(elev=30, azim=45)
    
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_zlabel('Z Coordinate (m)')
    ax.set_title(f'3D Cloud Tracking: Cloud {cloud_t0.cloud_id}')
    
    # Fix the legend to avoid duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
    
    status = "SUCCESS" if match_found else "FAILURE"
    plt.suptitle(f"Match attempt from t={cloud_t0.timestep} to t={cloud_field_t1.timestep}: {status}", fontsize=16)
    
    # Improve layout to make room for the navigation message
    plt.subplots_adjust(bottom=0.05)
    
    # Return the figure to allow further manipulation if needed
    return fig, ax


def main():
    # --- Configuration (should match your main.py) ---
    base_file_path = '/Users/jure/PhD/coding/RICO_1hr/'
    file_name = {
        'l': 'rico.l.nc', 'u': 'rico.u.nc', 'v': 'rico.v.nc',
        'w': 'rico.w.nc', 'p': 'rico.p.nc', 't': 'rico.t.nc', 'q': 'rico.q.nc'
    }
    config = {
        'min_size': 10, 'l_condition': 0.001, 'w_condition': 0.0,
        'w_switch': False, 'timestep_duration': 60, 'distance_threshold': 0,
        'plot_switch': False, 'horizontal_resolution': 25.0,
        'switch_wind_drift': True, 'switch_vertical_drift': True,
        'cloud_base_altitude': 700,
    }

    # --- Debug Parameters ---
    TIMESTEP_T0 = 3  # The first timestep to load
    CLOUD_LABEL_TO_DEBUG = 5  # The label of the cloud you want to investigate

    # --- Script ---
    # Initialize the tracker
    tracker = CloudTracker(config)

    # Load data for the two consecutive timesteps
    print(f"Loading data for timestep {TIMESTEP_T0}...")
    cloud_field_t0 = load_cloud_field_from_file(base_file_path, file_name, TIMESTEP_T0, config)
    
    print(f"Loading data for timestep {TIMESTEP_T0 + 1}...")
    cloud_field_t1 = load_cloud_field_from_file(base_file_path, file_name, TIMESTEP_T0 + 1, config)

    # Find the specific cloud to debug
    cloud_id_to_find = f"{TIMESTEP_T0}-{CLOUD_LABEL_TO_DEBUG}"
    if cloud_id_to_find not in cloud_field_t0.clouds:
        print(f"Error: Cloud with ID '{cloud_id_to_find}' not found in timestep {TIMESTEP_T0}.")
        print(f"Available cloud labels are: {[c.split('-')[1] for c in cloud_field_t0.clouds.keys()]}")
        return

    cloud_to_debug = cloud_field_t0.clouds[cloud_id_to_find]
    
    # Set tracker dimensions (needed for is_match)
    tracker.xt = cloud_field_t0.xt
    tracker.yt = cloud_field_t0.yt
    tracker.zt = cloud_field_t0.zt
    tracker.domain_size_x = (tracker.xt[-1] - tracker.xt[0]) + config['horizontal_resolution']
    tracker.domain_size_y = (tracker.yt[-1] - tracker.yt[0]) + config['horizontal_resolution']
    cloud_to_debug.is_active = True # Ensure it's considered for matching

    # Generate the visualisation
    fig, ax = visualise_match_attempt(cloud_to_debug, cloud_field_t1, tracker, config, tracker.xt, tracker.yt, tracker.zt)
    
    # Make sure the plot is displayed and blocks until closed
    plt.show(block=True)

if __name__ == "__main__":
    main()