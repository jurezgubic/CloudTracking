import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from netCDF4 import Dataset
import matplotlib.cm as cm
from matplotlib.widgets import Button

def visualize_cloud_tracks_3d():
    """Create an interactive 3D visualization of cloud tracks with rotation controls"""
    # Load data
    nc_file = '../cloud_results.nc'
    dataset = Dataset(nc_file, 'r')
    
    # Load necessary variables
    x_centers = dataset.variables['location_x'][:]
    y_centers = dataset.variables['location_y'][:]
    z_centers = dataset.variables['location_z'][:]
    valid_tracks = dataset.variables['valid_track'][:]
    sizes = dataset.variables['size'][:]
    
    # Filter for complete lifecycle clouds with valid data
    valid_cloud_indices = (np.any(~np.isnan(x_centers), axis=1) & 
                          np.any(~np.isnan(y_centers), axis=1) &
                          np.any(~np.isnan(z_centers), axis=1) &
                          (valid_tracks == 1))
    active_clouds = np.sum(valid_cloud_indices)
    
    # Create colormap for clouds
    colors = cm.get_cmap('tab20', active_clouds)
    
    # Create figure with 3D axes
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Track which clouds are drawn
    drawn_clouds = []
    color_idx = 0
    
    # Plot each cloud track
    for cloud_idx in range(len(valid_cloud_indices)):
        if valid_cloud_indices[cloud_idx]:
            valid_idx = (~np.isnan(x_centers[cloud_idx, :]) & 
                         ~np.isnan(y_centers[cloud_idx, :]) &
                         ~np.isnan(z_centers[cloud_idx, :]))
            
            if np.any(valid_idx):
                x_track = x_centers[cloud_idx, valid_idx]
                y_track = y_centers[cloud_idx, valid_idx]
                z_track = z_centers[cloud_idx, valid_idx]
                
                # Get timesteps for this cloud
                timesteps = np.where(valid_idx)[0]
                
                # Get sizes for marker scaling
                cloud_sizes = sizes[cloud_idx, valid_idx]
                # Use sqrt scaling for better visual representation
                marker_sizes = 50 * np.sqrt(cloud_sizes) / np.sqrt(np.max(cloud_sizes) + 1e-10) + 20
                
                # Plot track line
                ax.plot(x_track, y_track, z_track, '-', 
                       color=colors(color_idx), 
                       alpha=0.7, 
                       label=f'Cloud {cloud_idx}')
                
                # Plot points with timestep labeling
                for i, (x, y, z, t, size) in enumerate(zip(x_track, y_track, z_track, 
                                                          timesteps, marker_sizes)):
                    ax.scatter(x, y, z, s=size, 
                             color=colors(color_idx),
                             edgecolor='black', 
                             alpha=0.8)
                    # Show timestep for start and end points
                    if i == 0 or i == len(x_track)-1:
                        ax.text(x, y, z, f't={t}', fontsize=8)
                
                drawn_clouds.append(cloud_idx)
                color_idx += 1
    
    # Setup plot
    ax.set_xlabel('X Location (m)')
    ax.set_ylabel('Y Location (m)')
    ax.set_zlabel('Z Location (m)')
    ax.set_title(f'3D Movement of Cloud Centers - {len(drawn_clouds)} Clouds')
    ax.grid(True)
    
    # Add legend if not too many clouds
    if len(drawn_clouds) <= 10:
        ax.legend(loc='upper right')
    
    # Add viewing angle buttons
    ax_top = plt.axes([0.85, 0.01, 0.1, 0.05])
    top_button = Button(ax_top, 'Top View')
    
    ax_side = plt.axes([0.85, 0.07, 0.1, 0.05])
    side_button = Button(ax_side, 'Side View')
    
    ax_reset = plt.axes([0.85, 0.13, 0.1, 0.05])
    reset_button = Button(ax_reset, 'Reset View')
    
    def top_view(event):
        ax.view_init(elev=90, azim=-90)
        fig.canvas.draw_idle()
        
    def side_view(event):
        ax.view_init(elev=0, azim=0)
        fig.canvas.draw_idle()
        
    def reset_view(event):
        ax.view_init(elev=30, azim=-60)
        fig.canvas.draw_idle()
    
    top_button.on_clicked(top_view)
    side_button.on_clicked(side_view)
    reset_button.on_clicked(reset_view)
    
    plt.tight_layout()
    print("3D visualization launched. Close the window to exit.")
    print("You can rotate the view by clicking and dragging with the mouse.")
    plt.show(block=True)  # Ensure the plot stays open
    
    dataset.close()
    return fig, ax

if __name__ == "__main__":
    visualize_cloud_tracks_3d()