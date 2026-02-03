import sys
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arrow
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from scipy.spatial import ConvexHull, Delaunay
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.widgets import Button
from datetime import datetime
# Import plotly for interactive HTML export (install with: pip install plotly)
import plotly.graph_objects as go
import plotly.io as pio

# Add project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_management import load_cloud_field_from_file
from lib.cloudtracker import CloudTracker


def create_cloud_surface(cloud_points, ax, color, alpha=0.3, edge_color=None, label=None):
    """
    Create a 3D surface visualization of a cloud using its points.
    
    Args:
        cloud_points: NumPy array of shape (n, 3) containing x, y, z coordinates
        ax: Matplotlib 3D axes object
        color: Color for the surface
        alpha: Transparency level (0-1)
        edge_color: Color for edges (if None, uses a lighter version of color)
        label: Label for the legend
        
    Returns:
        success: Boolean indicating if surface creation was successful
    """
    if len(cloud_points) < 4:
        return False
    
    if edge_color is None:
        # Create a lighter version of the color for edges
        edge_color = mcolors.to_rgba(color, alpha=1.0)
        
    try:
        # Create a convex hull representation of the cloud
        hull = ConvexHull(cloud_points)
        
        # Get simplices (triangles) from the hull
        simplices = hull.simplices
        
        # Create a Poly3DCollection for the hull faces
        faces = []
        for simplex in simplices:
            faces.append([cloud_points[s] for s in simplex])
        
        # Plot the cloud surface
        poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, 
                              edgecolor=edge_color, linewidths=0.5)
        ax.add_collection3d(poly)
        
        # Add this to the legend manually if label is provided
        if label:
            ax.plot([], [], linestyle="-", color=color, alpha=alpha, label=label)
            
        return True
        
    except Exception as e:
        print(f"Could not create cloud surface: {e}")
        return False


def get_points_from_cloud(cloud):
    """Extract and convert cloud points to numpy array."""
    if not cloud.points:
        return None
        
    x_coords = [point[0] for point in cloud.points]
    y_coords = [point[1] for point in cloud.points]
    z_coords = [point[2] for point in cloud.points]
    
    return np.column_stack((x_coords, y_coords, z_coords))


def export_to_plotly(cloud_fields, match_chain, all_cloud_data):
    """
    Export the cloud tracking visualization as an interactive HTML file using Plotly.
    
    Args:
        cloud_fields: List of CloudField objects
        match_chain: List of cloud IDs that form the tracking chain
        all_cloud_data: Dictionary with cloud visualization data
        
    Returns:
        None (saves HTML file)
    """
    fig = go.Figure()
    
    # Add tracked clouds
    for i, cloud_id in enumerate(match_chain):
        timestep = int(cloud_id.split('-')[0])
        points = all_cloud_data[cloud_id]['points']
        color = all_cloud_data[cloud_id]['color']
        
        # Add cloud points
        fig.add_trace(go.Mesh3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            color=color,
            opacity=0.5,
            name=f"Cloud {cloud_id} (t+{i})"
        ))
        
        # Add cloud center
        center = all_cloud_data[cloud_id]['center']
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode='markers',
            marker=dict(size=10, color=color, opacity=1),
            name=f"Center {cloud_id} (t+{i})"
        ))
    
    # Add nearby clouds (if available)
    for cloud_id, data in all_cloud_data.items():
        if 'nearby' in data and data['nearby']:
            points = data['points']
            fig.add_trace(go.Mesh3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                color='grey',
                opacity=0.2,
                name=f"Nearby {cloud_id}"
            ))
    
    # Add movement lines
    for i in range(len(match_chain)-1):
        start_center = all_cloud_data[match_chain[i]]['center']
        end_center = all_cloud_data[match_chain[i+1]]['center']
        
        fig.add_trace(go.Scatter3d(
            x=[start_center[0], end_center[0]],
            y=[start_center[1], end_center[1]],
            z=[start_center[2], end_center[2]],
            mode='lines',
            line=dict(color='black', width=2, dash='dash'),
            name=f"Movement {match_chain[i]} → {match_chain[i+1]}"
        ))
    
    # Set layout
    fig.update_layout(
        title=f"Cloud Tracking Visualization: {' → '.join(match_chain)}",
        scene=dict(
            xaxis_title='X Coordinate (m)',
            yaxis_title='Y Coordinate (m)',
            zaxis_title='Z Coordinate (m)',
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Save as HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"cloud_tracking_{match_chain[0]}_{timestamp}.html"
    pio.write_html(fig, filename)
    print(f"Interactive visualization saved as: {filename}")
    
    return filename


def visualise_multi_timestep(cloud_fields, tracker, config, timesteps=4, max_nearby_clouds=5):
    """
    Visualise cloud tracking across multiple timesteps with nearby clouds.
    
    Args:
        cloud_fields: List of CloudField objects for each timestep
        tracker: The CloudTracker instance
        config: Configuration dictionary
        timesteps: Number of timesteps to visualize
        max_nearby_clouds: Maximum number of nearby clouds to show at each timestep
        
    Returns:
        fig, ax: Figure and axes objects
    """
    # Setup the figure and axes
    plt.ion()
    fig = plt.figure(figsize=(16, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add interactivity instructions
    plt.figtext(0.5, 0.01, 
                "Interactive Controls: Left-click drag to rotate | Right-click drag to zoom | Middle-click drag to pan",
                ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Colors for the tracked cloud at different timesteps
    cloud_colors = ['green', '#88c999', '#ffdd55', '#ff9955', '#ff7733', '#ff5511']
    
    # Track the cloud across timesteps
    tracked_cloud = cloud_fields[0].clouds[f"{tracker.current_timestep}-{tracker.cloud_to_track}"]
    
    # Lists to store min/max coordinates for view limits
    all_x, all_y, all_z = [], [], []
    
    # Plot the initial cloud
    points = get_points_from_cloud(tracked_cloud)
    if points is not None:
        create_cloud_surface(points, ax, cloud_colors[0], label=f"Cloud {tracked_cloud.cloud_id} (t)")
        
        # Add cloud center
        loc = tracked_cloud.location
        ax.scatter([loc[0]], [loc[1]], [loc[2]], c='darkgreen', marker='o', 
                  s=150, alpha=1.0, label=f'Center {tracked_cloud.cloud_id} (t)')
                  
        # Update coordinate limits
        all_x.extend([p[0] for p in tracked_cloud.points])
        all_y.extend([p[1] for p in tracked_cloud.points])
        all_z.extend([p[2] for p in tracked_cloud.points])
    
    # Previous location for drawing movement lines
    prev_loc = tracked_cloud.location
    current_cloud = tracked_cloud
    
    # Track the cloud through subsequent timesteps
    match_chain = [tracked_cloud.cloud_id]
    
    for t in range(1, min(timesteps, len(cloud_fields))):
        match_found = False
        
        # Find the matched cloud in the next timestep
        for cloud_id, cloud in cloud_fields[t].clouds.items():
            if tracker.is_match(cloud, current_cloud):
                # Plot the matched cloud
                points = get_points_from_cloud(cloud)
                if points is not None:
                    color_idx = min(t, len(cloud_colors)-1)
                    create_cloud_surface(
                        points, ax, cloud_colors[color_idx], 
                        label=f"Cloud {cloud.cloud_id} (t+{t})"
                    )
                    
                    # Add cloud center
                    loc = cloud.location
                    ax.scatter([loc[0]], [loc[1]], [loc[2]], 
                              c=mcolors.to_rgba(cloud_colors[color_idx], 0.8), marker='o', 
                              s=150, alpha=1.0, label=f'Center {cloud.cloud_id} (t+{t})')
                    
                    # Draw line showing cloud movement
                    ax.plot([prev_loc[0], loc[0]], [prev_loc[1], loc[1]], [prev_loc[2], loc[2]], 
                           'k--', alpha=0.8, linewidth=2)
                    
                    # Update for next iteration
                    prev_loc = loc
                    current_cloud = cloud
                    match_found = True
                    match_chain.append(cloud.cloud_id)
                    
                    # Update coordinate limits
                    all_x.extend([p[0] for p in cloud.points])
                    all_y.extend([p[1] for p in cloud.points])
                    all_z.extend([p[2] for p in cloud.points])
                    
                    break
        
        # If no match found, break the chain
        if not match_found:
            print(f"No match found at timestep t+{t}")
            break
            
        # Find and plot nearby clouds (in grey)
        loc = current_cloud.location
        nearby_clouds = []
        
        for cloud_id, cloud in cloud_fields[t].clouds.items():
            # Skip the tracked cloud
            if cloud_id == current_cloud.cloud_id:
                continue
                
            # Calculate distance to tracked cloud
            cloud_loc = cloud.location
            distance = np.sqrt(
                (loc[0] - cloud_loc[0])**2 + 
                (loc[1] - cloud_loc[1])**2 + 
                (loc[2] - cloud_loc[2])**2
            )
            
            # Add to nearby clouds if within range
            if distance < 10000:  # 10km radius
                nearby_clouds.append((cloud, distance))
        
        # Sort by distance and take the closest few
        nearby_clouds.sort(key=lambda x: x[1])
        for cloud, distance in nearby_clouds[:max_nearby_clouds]:
            points = get_points_from_cloud(cloud)
            if points is not None:
                # Create semi-transparent grey surface
                success = create_cloud_surface(
                    points, ax, 'grey', alpha=0.15, 
                    edge_color='lightgrey', 
                    label=f"Nearby Cloud {cloud.cloud_id}" if len(nearby_clouds) <= 3 else None
                )
                
                # If surface creation failed, fall back to scatter plot
                if not success:
                    x = [p[0] for p in cloud.points]
                    y = [p[1] for p in cloud.points]
                    z = [p[2] for p in cloud.points]
                    ax.scatter(x, y, z, c='grey', marker='.', s=10, alpha=0.2)
    
    # Set view limits with padding
    if all_x and all_y and all_z:
        padding = 1000  # meters
        ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
        ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
        ax.set_zlim(min(all_z) - padding, max(all_z) + padding)
    
    # Set better 3D viewing angle
    ax.view_init(elev=30, azim=45)
    
    # Add labels and title
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_zlabel('Z Coordinate (m)')
    ax.set_title(f'Multi-timestep Cloud Tracking: {match_chain[0]}')
    
    # Fix the legend to avoid duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
    
    # Status summary
    plt.suptitle(
        f"Cloud tracking across {len(match_chain)} timesteps: {' → '.join(match_chain)}", 
        fontsize=16
    )
    
    # Improve layout
    plt.subplots_adjust(bottom=0.05)
    
    return fig, ax
def main():
    """Run the multi-timestep tracking visualization with interactive display only."""
    # Configuration (should match your main.py)
    base_file_path = '/Users/jure/PhD/coding/RICO_1hr/'
    file_name = {
        'l': 'rico.l.nc', 'u': 'rico.u.nc', 'v': 'rico.v.nc',
        'w': 'rico.w.nc', 'p': 'rico.p.nc', 't': 'rico.t.nc', 'q': 'rico.q.nc',
        'r': 'rico.r.nc'  # Rain water mixing ratio (optional)
    }
    config = {
        'min_size': 10, 'l_condition': 0.001, 'w_condition': 0.0,
        'w_switch': False, 'timestep_duration': 60, 'distance_threshold': 0,
        'plot_switch': False, 'horizontal_resolution': 25.0,
        'switch_wind_drift': True, 'switch_vertical_drift': True,
        'cloud_base_altitude': 700,
    }

    # Debug Parameters
    START_TIMESTEP = 3
    CLOUD_TO_TRACK = 2
    NUM_TIMESTEPS = 4
    
    # Initialize the tracker with additional properties
    tracker = CloudTracker(config)
    tracker.current_timestep = START_TIMESTEP
    tracker.cloud_to_track = CLOUD_TO_TRACK
    
    # Load data for multiple timesteps
    cloud_fields = []
    
    print(f"Loading data for {NUM_TIMESTEPS} timesteps starting at {START_TIMESTEP}...")
    for t in range(NUM_TIMESTEPS):
        timestep = START_TIMESTEP + t
        print(f"Loading timestep {timestep}...")
        cloud_field = load_cloud_field_from_file(base_file_path, file_name, timestep, config)
        cloud_fields.append(cloud_field)
        
        # Set tracker dimensions from the first timestep (needed for is_match)
        if t == 0:
            tracker.xt = cloud_field.xt
            tracker.yt = cloud_field.yt
            tracker.zt = cloud_field.zt
            tracker.domain_size_x = (tracker.xt[-1] - tracker.xt[0]) + config['horizontal_resolution']
            tracker.domain_size_y = (tracker.yt[-1] - tracker.yt[0]) + config['horizontal_resolution']
            
            # Verify the cloud to track exists
            cloud_id_to_find = f"{START_TIMESTEP}-{CLOUD_TO_TRACK}"
            if cloud_id_to_find not in cloud_field.clouds:
                print(f"Error: Cloud with ID '{cloud_id_to_find}' not found in timestep {START_TIMESTEP}.")
                print(f"Available cloud labels are: {[c.split('-')[1] for c in cloud_field.clouds.keys()]}")
                return
    
    # Generate the visualization
    fig, ax = visualise_multi_timestep(cloud_fields, tracker, config, NUM_TIMESTEPS)
    
    # Display the matplotlib visualization and keep it open until closed by user
    plt.show(block=True)


if __name__ == "__main__":
    main()