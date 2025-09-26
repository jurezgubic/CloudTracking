import argparse
import os
from analysis.track_statistics import compute_statistics, visualise_statistics
from analysis.cloud_lifecycle_visualisation import visualise_cloud_lifecycles

def main():
    parser = argparse.ArgumentParser(description="Analyze cloud tracking results.")
    parser.add_argument("netcdf_file", help="Path to NetCDF file with cloud tracking data")
    parser.add_argument("--output-dir", default="./analysis_output", help="Output directory for visualisations")
    parser.add_argument("--min-timesteps", type=int, default=3, help="Minimum cloud lifetime in timesteps")
    parser.add_argument("--min-size", type=int, default=10, help="Minimum cloud size (grid points)")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Analyzing cloud tracks from {args.netcdf_file}...")
    print(f"Filtering for: complete lifecycles, ≥{args.min_timesteps} timesteps, size ≥{args.min_size}")
    
    # Generate track statistics with the same filters
    print("Computing track statistics...")
    stats = compute_statistics(args.netcdf_file, min_timesteps=args.min_timesteps, min_size=args.min_size)
    visualise_statistics(stats, os.path.join(args.output_dir, "track_statistics.png"))
    
    # Generate cloud lifecycle visualisation with the same filters
    print("Creating cloud lifecycle visualisation...")
    visualise_cloud_lifecycles(
        args.netcdf_file, 
        os.path.join(args.output_dir, "cloud_lifecycles.png"),
        min_valid_timesteps=args.min_timesteps,
        min_size_threshold=args.min_size,
        include_partial=False  # Only include complete lifecycle tracks
    )
    
    print(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()