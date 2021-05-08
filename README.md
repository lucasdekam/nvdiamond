# Characterizing a permanent magnet using NV center electron spin
This is the code and data for a TU Delft research practicum. The contents are as follows:
* Raw data can be found in the rawdata folder. For the varying_distance dataset, the folder names indicate the reading on the magnet scale. 63 mm is the reading on the magnet scale at the initial distance of ~5 cm between the diamond crystal and the magnet center, and this distance was decreased every step. 
* (Manually) processed data is stored in the .txt files. The first column is the distance or angle, the next 8 columns are the peak center frequencies, and the last 8 columns are the uncertainties for each peak frequency.
* Code files that generate the plots in the report from the processed data files: b_direction_esrfit.py, b_magnitude_angle.py, b_magnitude_distance.py
* Code files that generate plots for the theory section: assignments.py and plotfieldlines.py
