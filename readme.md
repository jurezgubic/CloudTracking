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
