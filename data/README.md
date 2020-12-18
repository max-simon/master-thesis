# Data

The data is located in `/nfs/kryo/work/maxsimon/data/pactcs30` for pactcs30 (or mid-resolution) and in `/nfs/kryo/work/maxsimon/data/pactcs15` for pactcs15 (or high-resolution). Because the runs are compared, the structure of the two folders is identical.

## Folder structure

- `grid.nc`: Grid file of the run
- `grid.npz`: Additional grid data, i.e. definition of analysis doman and distance map. Created by [grids.ipynb](../notebooks/grids.ipynb).
- `d_*.nc`: Average files from ROMS (bidaily averages).
- `z`
  - `z_d_*.nc`: ROMS output interpolated to constant depths. Created by `zslice`.
  - `z_vel.nc`: Horizontal velocity interpolated to rho-grid. Created by [interp.py](../scripts/interp.py).
  - `vort.nc`: Relative vorticity. Created by `vort`.
  - `z_levels.nc`: Depth and thickness of depth layers. Created by [zlevel.py](../scripts/zlevel.py).
  - `eke.nc`: Surface eddy kinetic energy. Created by [eke.py](../scripts/eke.py).
- `nitro`
  - `z_d_*.nc`: DON and NH4 interpolated to constant depths. The same depth levels as for `z/z_d_*.nc` were used. Created by `zslice`.
  - `X_prime.nc`: Anomalies for X = NO3, NH4, DON, w and v. All files refer to constant depths and were created by [primes.py](../scripts/primes.py). `total_prime.nc` is the sum of NO3, NH4 and DON (total nitrate). The files are used to calculate the eddy-induced nitrate flux in [eddy-quenching.ipynb](../notebooks/eddy-quenching.ipynb).
- `climatologies` (all files were created by [climatology.py](../scripts/climatology.py) with `--days-around 1` to interpolate missing days in bidaily output).
  - `z_data-1d.nc`: Climatology of `z/z_d_*.nc`.
  - `z_vel-1d.nc`: Climatology of `z/z_vel.nc`.
  - `z_vort-1d.nc`: Climatology of `z/vort.nc`.
  - `z_nitro-1d.nc`: Climatology of `nitro/z_d_*.nc`.
  - `zeta-1d.nc`: Climatology of `d_*.nc` (variable zeta)
  - `hbls-1d.nc`: Climatology of `d_*.nc` (variable hbls)
  - `smooth`
    - `clim-0deg-X_b.nc`: Soft link to correct climatology file based on variable name X
- `ssh`
  - `ssh.nc`: Data for zeta from `d_*.nc` combined in one file. This is the required input format for Faghmous Eddy Detection Algorithm.
  - `raw`: SSH data in MATLAB format (output of Faghmous Algorithm).
  - `eddies`: Eddy detection results in MATLAB format (output of Faghmous Algorithm).
  - `tracks`: Tracking results in MATLAB format (output of Faghmous Algorithm).
  - `eddies-00000.nc`: Transformed eddy detection and tracking results. Created by [eddies.py](../scripts/eddies.py).
  - `final_eddy_attr_map_cyc-int-ampl.nc`: Spatial maps of eddy attributes (see [description](../scripts/README.md)). Created by [eddies_attribute_map.py](../scripts/eddies_attribute_map.py).
  - `ano`
    - `final_eke_surface-00000.nc`: Surface eddy kinetic energy anomalies in mesoscale eddies. Created by [eddies_anomaly.py](../scripts/eddies_anomaly.py).
    - `final_rho_18to25-00000.nc`: Density anomalies in mesoscale eddies, averaged from 80m to 120m. Created by [eddies_anomaly.py](../scripts/eddies_anomaly.py).
    - `final_rho_7to33-00000.nc`: Density anomalies in mesoscale eddies, averaged from 20m to 200m. Created by [eddies_anomaly.py](../scripts/eddies_anomaly.py).
- `ssh_152` (Only in pactcs30. The folder contains the eddy detection results for pactcs15 interpolated to pactcs30. It has the same structure as `ssh`.)
- `composites`:
  - `r35x750x7-30d-*`: Composite data for different variables. The used eddies have a minimum radius of 35km and a minimum lifetime of 7 days. Created by [composites.py](../scripts/composites.py).
