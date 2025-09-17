# LES Cloud Tracking and Analysis code

## Description and Aim
This project provides Python code designed for the processing, tracking, and analysing of LES cloud data. The aim of is to facilitate easy and efficient analysis of cloud dynamics and properties over time.

See [Output Analysis Notebook](analysis/output_analysis.ipynb) for example usage. 

## Features
- **Cloud Analysis**: Tracks and analyses cloud data, including cloud volume, surface area, time, location, maximum height, cloud base size, maximum vertical velocity and maximum vertical velocity at the cloud base (currently).
- **Data Loading and Management**: Manages large datasets using NetCDF.
- **Dynamic Tracking**: Accounts for wind drift between time steps.
- **Biperiodicity**: Works for biperiodic LES domains.
- **Structure Identification Flexibility**: Allows for user choice of cloud identification thresholds, including liquid water, vertical velocity, and cloud size (currently).


## Summary of modules
0. **input_field_analysis.ipynb**: Inspects basic characteristics of input data.
1. **main.py**: Directs the workflow, initialising processes and managing other modules.
2. **data_management.py**: Manages data loading and preprocessing of NetCDF data.
3. **cloud.py**: Creates a cloud object that is passed around in other modules.
4. **cloudfield.py**: Identifies cloud objects in the loaded dataset
5. **cloudtracker.py**: Tracks and analyses temporal changes of clouds in cloudield.
6. **netcdf_writer.py**: Writes data to NetCDF files. 
7. **output_analysis_general.ipynb**: Basic analysis for RICO run.
8. **output_analysis_vertical_profiles.ipynb**: Analysis of vertical profiles. 


## How to use
```bash
git clone https://github.com/jurezgubic/CloudTracker.git
cd CloudTracker
pip install -r requirements.txt
```

##  Running this code 
### On its own
1. Set config file in main.py
2. Set correct paths to LES data.
3. Run with `python main.py`
4. To inspect output run analysis/output_analysis.ipynb notebook.


### With memory profiler
- Run with `python -m memory_profiler script_name.py`
- Or`mprof run main.py`. Then plot using `mprof plot`
- Warning: Do NOT run for the entire dataset (far too expensive). 

### Basic Output Visualisation
Use the following command to visualise cloud lifecycles, active clouds per timestep, and generate lifetime and size bar charts:

```bash
python analyse_clouds.py cloud_results.nc
```

## Visualisation & Analysis Toolkit

All tools consume the NetCDF produced by `main.py` (default: `cloud_results.nc`).  
By default, analysis scripts exclude partial (tainted) tracks (those with `valid_track=0`).

### 1. Quick Batch Analysis (stats + lifecycle plot)
Wrapper that applies consistent filters (complete lifecycles, min timesteps, min size):
```bash
python analyse_clouds.py cloud_results.nc \
  --output-dir ./analysis_output \
  --min-timesteps 3 \
  --min-size 10
```
Outputs:
- `track_statistics.png`
- `cloud_lifecycles.png`

### 2. Track Statistics (programmatic)
```bash
python -c "from analysis.track_statistics import compute_statistics, visualise_statistics; \
s=compute_statistics('cloud_results.nc', min_timesteps=3, min_size=10); \
visualise_statistics(s, 'analysis_output/track_statistics.png')"
```
Reports (for filtered complete tracks): counts, merge events, active clouds per timestep, lifetime distribution, size evolution.

### 3. Lifecycle Timeline Plot
```bash
python -c "from analysis.cloud_lifecycle_visualisation import visualise_cloud_lifecycles; \
visualise_cloud_lifecycles('cloud_results.nc','analysis_output/cloud_lifecycles.png', \
max_tracks=40, min_valid_timesteps=3, min_size_threshold=10, include_partial=False)"
```
`include_partial=False` is the recommended default.

### 4. 3D Track Explorer
Interactive centroid trajectories; marker size ∝ sqrt(cloud size):
```bash
python analysis/cloud_3d_visualizer.py
```
Requires an interactive Matplotlib backend (e.g. TkAgg). Large numbers of tracks may slow rendering.

### 5. High-Base Cloud Surface Visualisation (Experimental)
File: `analysis/non_LCL_cloud_visual.py`  
Status: INCOMPLETE. To become usable, the following must be implemented:
- Populate NetCDF reads inside `find_high_initiated_clouds()`.
- Build list of candidate clouds (filter by first valid height ≥ `height_threshold` & lifetime ≥ `min_timesteps`).
- Complete early returns where placeholders exist in `extract_cloud_surface` (np, os, plt, skimage imports missing).
Mark remains experimental until these are done:
```bash
# (After completion)
python analysis/non_LCL_cloud_visual.py
```

### 6. Debug / Matching Helpers
`utils/plotting_utils.py` provides:
- `visualize_points` / `visualize_points_plotly`: compare prior, drifted, and current surface point clouds (used during matching).
- `plot_labeled_regions`: save 2‑D labeled slices (enable by setting `config['plot_switch']=True` in `main.py`).


### 7. Partial (Tainted) Tracks
Definition (intended design):
- Track starting at first processed timestep or still active at final timestep have a potentially incomplete lifecycle.
- Any track inheriting from a partial (via merge) may also be marked tainted.
Default: exclude these from all analysis plots/statistics. To include: set `include_partial=True` in lifecycle call (not exposed in `analyse_clouds.py` yet—could be added).

### 8. Performance Notes
- Global KD-tree of surface points built in `CloudField.build_global_surface_kdtree()` accelerates overlap queries.
- Centroid pre-filtering (`pre_filter_cloud_matches`) reduces expensive surface overlap checks; can disable with config flag `use_pre_filtering=False`.
- When pre-filtering is enabled, you can control the fallback to a full-domain search if no match candidates are found with `switch_prefilter_fallback` (default `True`). Set `switch_prefilter_fallback=False` to skip the fallback to all and move on. Saves some time.
- For large domains consider future migration to xarray + dask (not yet implemented).

### 9. Troubleshooting
| Issue | Cause | Action |
|-------|-------|--------|
| No tracks in plots | All filtered out (too strict thresholds or all tainted) | Lower `--min-timesteps` / `--min-size`; inspect `valid_track` values |
| Stats mostly empty | Placeholder code in statistics script not finished | Complete loops in `analysis/track_statistics.py` |
| 3D viewer slow | Too many active tracks | Subset timesteps or add filtering logic |
