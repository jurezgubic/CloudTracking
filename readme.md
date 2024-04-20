3d Cloud tracking code


# LES Cloud Data Tracking and Analysis code

## Description and Aim
This project provides Python code designed for the processing, analysis, and tracking of LES cloud data. The aim of is to facilitate easy and efficient analysis of cloud dynamics and properties over time.

## Features
- **Data Loading and Management**: Manage large sets of cloud data efficiently.
- **Cloud Analysis**: Perform detailed analysis on cloud properties and dynamics.
- **NetCDF Integration**: Utilize NetCDF files for robust data storage and retrieval.
- **Cloud Tracking**: Track changes in cloud fields over time to observe trends and patterns.

## Summary of modules
0. **input_field_analysis.ipynb**: Inspects basic characteristics of input data.
1. **main.py**: Directs the workflow, initialising processes and managing other modules.
2. **data_management.py**: Manages data loading and preprocessing of NetCDF data.
3. **cloud.py**: Creates a cloud object that is passed around in other modules.
4. **cloudfield.py**: Identifies cloud objects in the loaded dataset
5. **cloudtracker.py**: Tracks and analyses temporal changes of clouds in cloudield.
6. **netcdf_writer.py**: Writes data to NetCDF files. 
7. **output_analysis.ipynb**: Basic analysis for RICO run. 


## Installation

```bash
git clone https://github.com/yourgithubusername/cloud-data-analysis-toolkit.git
cd cloud-data-analysis-toolkit
pip install -r requirements.txt
```

##  Running this code 
### On its own
1. Set config file in main.py
2. Set correct path to LES data.
3. Run with "python main.py"
4. To inspect output run analysis/output_analysis.ipynb notebook.


### With memory profiler
- Uncomment profiler decorators 
- Or "mprof run main.py" and then plot using "mprof plot"
- Warning: Do NOT run for the entire dataset (far too expensive). 

