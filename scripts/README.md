# Scripts


## Tools

- [interp.py](interp.py): This is a wrapper for `xesmf`. Using this script, ROMS output can be interpolated from one grid to another (different grid name as well as different grid file). The script can be used from command line (see help).

- [slice.py](slice.py): The script can be used to slice a full dataset on rho-grid, i.e. u-, v-, w- and psi-grids are sliced accordingly. When using from command line, the script outputs the full `ncks` command (see help).

- [zlevel.py](zlevel.py): Calculate the depth of terrain-following vertical coordinates (ask Eike Köhn for original script). The script was extended to also calculate the thickness of the layers and to output the data to a netCDF dataset. The script can be used from command line (see help).

- [resolution.py](resolution.py): Calculate the nominal resolution of a grid (see [CMIP6 definition](https://github.com/PCMDI/nominal_resolution)). The script can be used from command line, but the paths have to be adjusted.


## Data Processing

- [climatology.py](climatology.py): The script creates climatologies for ROMS output. The script can be used from command line (see help). The option `--days-around` applies a rolling mean for each doy. You might want to set this to 1 to interpolate missing days in bidaily output.

- [eddies.py](eddies.py): The script transforms the output of eddy detections to a more suitable netCDF4 dataset. The data is supplemented by the distance to coast, the travelled distance, amplitude and intensity of an eddy. In addition, spatial maps are created for each timestep containing eddy and track indices. The script can be used from command line (see help).

- [eke.py](eke.py): The script calculates EKE and related variables for ROMS output. The script can be used from command line (see help).

- [eddies_anomaly.py](eddies_anomaly.py): The script calculates statistics of a value/anomaly associated with mesoscale eddies. The script creates Hovmöller data for the value/anomaly in cyclones, anticyclones and background seperately. The script can be used from command line (see help).

- [composites.py](composites.py): The script generates composites of mesoscale eddies. The eddies can be filtered based on their attributes. The data is interpolated to a fixed grid size, but not averaged. This allows for postprocessing by `CompositeManager` provided in `composite_tools`. In order to reduce storage, filters which are applied for all analyses should be applied here, additional filters in postprocessing. _WARNING:_ before using with different datasets read `#TODO` comments as some things are still hardcoded.

- [primes.py](primes.py): The script calculates anomalies for ROMS output. The calculation is parallelized. Other scripts calculate anomalies on the fly, but explicitly calculating them might be useful for fluxes. The script can be used from command line (see help).

- [eddies_attribute_map.py](eddies_attribute_map.py): The script creates spatial maps where the area of each detected eddy is filled with an attribute (amplitude, polarity, intensity, eddy index, track index) of the eddy instance. This is useful when calculating intersections or when calculating Hovmöller data of these attributes. The script can be used from command line (see help).


## Utils

- [dataset.py](dataset.py): The script contains utility functions to open datasets in `xarray`. This includes fixes for decoding dates, an automatic use of `xarray.mfdataset` and an integration into `ArgumentParser`.

- [romsrun.py](romsrun.py): The script provides a class for handling multiple data files belonging to the same run. It does consistency checks when loading different sources, allows for an easy access of the data and simplifies the creation of iterator objects. In addition, it wraps functions for plotting and animations as well as for calculating PSDs.

- [plot.py](plot.py): The script provides a powerful function to plot 2D data on a curvilinear grid. See the function signature for the different options. In addition, a function to plot 3D data on a regular grid is provided. This is useful for visualizing eddy composites with vertical dimension.

- [composite_tools.py](composite_tools.py): The script provides a class for an easy access and advanced filtering of eddy composites (data created with `composites.py`). For an explanation of the different options to filter composite data see function `__getitem__` in `CompositeManager`. In addition, wrappers to plotting functions for 2D and 3D composite data are provided. _WARNING:_ before using with different datasets read `#TODO` comments as some things are still hardcoded.

- [utils.py](utils.py): The script contains several utility functions, e.g. for calculating area maps, parsing and manipulating time strings, getting dimension info for variables, selecting day-of-years, getting triangular weights, calculting rolling means, caching results and propagating errors.

- [cmap.py](cmap.py): The script emulates color schema from [Gruber et al (2011)](https://www.nature.com/articles/ngeo1273) and provides a function to convert continous color schemes to stepwise schemes.

- [psd.py](psd.py): The script provides functions to calculate (complex) PSDs on curvilinear data and functions to visualize the results.



# Running Eddy Detection Algorithm

The eddy detection algorithm of [Faghmous et al (2015)](https://www.nature.com/articles/sdata201528) with modifications for curvilinear grids implemented by [Lovecchio et al (2018)](https://bg.copernicus.org/articles/15/5061/2018/) was used. The code can be found in `/home/lelisa/Scripts/Matlab/EddyTracking_NEW/Elisa_FaghmousBased_TrackingCode`. In order to run it on different data, some changes are required:
- In `complete_run_EL.m`, adjust input paths (`directory`, `grid_file`, `ssh_file`) and output paths (`ssh_path`, `eddies_save_path`, `tracks_save_path`)
- In `complete_run_EL.m`, adjust `box_bounds` to match the new domain
- If you run into index errors, the domain is probably too small. This can be fixed by reducing the dimensions for some placeholders. In the original code, the dimension is X = 200. X has to be smaller than your grid dimension. To change X _consistently_, the following changes are required:
  - in `eddyscan/lib/bottom_up_single_EL.m`:
    - line 71: `extrema = [zeros(size(extrema, 1), <X>), extrema, zeros(size(extrema, 1), <X>)];`
    - line 72: `sshExtended = [ssh_data(:, end-<X-1>:end), ssh_data(:, :), ssh_data(:, 1:<X>)];`
    - line 76: `extrema(:, 1:<X>) = origExtrema(:, end-<X-1>:end);`
    - line 77: `extrema(:, end-<X-1>:end) = origExtrema(:, 1:<X>);`
  - in `thresholdBU_EL.m`:
    - line 146: `[elat, elon] = weighted_centroid(cyc_ssh(:, <X+1>:end-X), stats.PixelList, stats.PixelIdxList, R);`
    - line 148: `[elat, elon] = weighted_centroid_irregular_grid_EL(cyc_ssh(:, <X+1>:end-X), stats.PixelList, stats.PixelIdxList, lat, lon, wm, wn);` 
