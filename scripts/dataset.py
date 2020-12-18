#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


import sys
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')

import xarray as xr
import numpy as np
from datetime import timedelta as tdelta
from datetime import datetime
import argparse

from romstools.utils import parse_slice
from romstools.slice import slice_on_rho_grid

xr.set_options(keep_attrs=True) # this is required to keep the attributes when modifying time!


def get_drop_except_fn(*keep_vars):
        """
        Drop all but some variables in a netCDF file
        """
        def drop_except(ds):
                # do nothing when nothing to drop and avoid dropping time
                if len(keep_vars) == 0 or (len(keep_vars) == 1 and keep_vars[0] == 'time'):
                        return ds
                # xarray only allows to specify what to drop, but not which to keep. So we need
                # to invert this set
                drop_vars = []
                for v in ds.variables:
                        if v not in keep_vars:
                                drop_vars.append(v)
                return ds.drop_vars(drop_vars)
        return drop_except


def open_glob_dataset(data_files, keep_vars=[], time_slice=None):
        """
        Open a Multifile Dataset with xarray using a glob expression.
        """
        sm_ds = None
        # if a list of files or a star in name, use xarray.mfdataset
        if type(data_files) == list or '*' in data_files:
                sm_ds = xr.open_mfdataset(
                        data_files, 
                        decode_times=False, 
                        combine="nested", 
                        parallel=True, 
                        concat_dim='time',  # concatenate on time
                        preprocess=get_drop_except_fn(*keep_vars)  # drop all values except keep_vars
                )
                # slice time
                if time_slice is not None:
                        sm_ds = sm_ds.isel(time=time_slice)
        # just open.
        else:
                drop_fn = get_drop_except_fn(*keep_vars)
                sm_ds = xr.open_dataset(data_files, decode_times=False)
                # slice time
                if time_slice is not None:
                        sm_ds = sm_ds.isel(time=time_slice)
                # drop all values except keep_vars
                sm_ds = drop_fn(sm_ds)
        return sm_ds


def set_time(ds, dt=tdelta(seconds=0)):
        """
        Fix time loading errors of xarray, i.e. it keeps the time attributes. Also allows for specifying an offset.
        """
        # get calendar and units attribute
        calendar = ds.time.attrs['calendar']
        units = ds.time.attrs['units']
        # decode times
        TimeCoder = xr.coding.times.CFDatetimeCoder()
        ds['time'] = ds.time.fillna(0) + dt.total_seconds()
        ds['time'] = xr.DataArray(TimeCoder.decode(ds.variables['time'], 'time'))

        ds = ds.set_coords(['time'])
        # restore attributes
        ds.time.attrs['calendar'] = calendar
        ds.time.attrs['units'] = units
        return ds


def open_dataset(input, variables=[], time_calendar=None, time_raw=None, time_units=None, time_offset=0, time_from=None, eta_rho_slice=None, xi_rho_slice=None, s_rho_slice=None):
        """
        Load dataset and grid file, overwrite calendar and units.
        """
        # open data
        dataset = open_glob_dataset(input, keep_vars=variables+['time'], time_slice=None)
        # slice data
        if eta_rho_slice is not None or xi_rho_slice is not None or s_rho_slice is not None:
                dataset = slice_on_rho_grid(dataset, eta_rho_slice=eta_rho_slice, xi_rho_slice=xi_rho_slice, s_rho_slice=s_rho_slice)
        # open another dataset to copy its time array to the opened dataset
        if time_from is not None:
                aux_ds = open_glob_dataset(time_from, keep_vars=['time'], time_slice=None)
                attrs = aux_ds.time.attrs
                dataset['time'] = xr.DataArray(aux_ds['time'].values, dims=('time',))
                dataset['time'].attrs = attrs
                aux_ds.close()

        # if no time to process, skip
        if 'time' not in dataset:
                return dataset

        # reset calendar and units
        if time_raw is not None:
                dataset['time'] = xr.DataArray(time_raw, dims=('time',))
        if time_calendar is not None:
                dataset.time.attrs['calendar'] = time_calendar
        if time_units is not None:
                dataset.time.attrs['units'] = time_units
        
        # initialize time
        dataset = set_time(dataset, dt=tdelta(seconds=time_offset))

        # calculate day of year
        time_attrs = {**dataset.time.attrs}
        doy = np.array([a.dayofyr for a in dataset.time.values]) - 1
        
        # create doy variable on data
        dataset['doy'] = xr.DataArray(doy, dims=('time',))
        dataset.time.attrs = time_attrs

        return dataset


def dataset_from_args(parser):
        """
        Add a group to input arguments for opening a dataset
        """
        # create a parsing group
        group = parser.add_argument_group("dataset")

        # add items to group
        group.add_argument("-i", "--input", type=str, nargs='+', help="Input path, glob is supported", required=True)
        
        # spatial slicing
        group.add_argument("--eta-rho", type=parse_slice, help="Slice input data at eta coordinates")
        group.add_argument("--xi-rho", type=parse_slice, help="Slice input data at xi coordinates")
        group.add_argument("--s-rho", type=parse_slice, help="Slice input data at s_rho coordinates (or depth if present)")
        
        ## Removed to reduce verbosity on --help. Add them to open_dataset when needed
        # group.add_argument("--time-units", type=str, help="Overwrite time units attribute")
        # group.add_argument("--time-calendar", type=str, help="Overwrite time calendar attribute")
        # group.add_argument("--time-from", type=str, help="Overwrite time data")
        
        group.add_argument("-v", "--variables", type=str, nargs="+", help="Choose variables to load", default=[])

        # create a loading function
        def load(args, variables=None, s_rho_slice=None):
                # variables and s_rho_slice can be overwritten
                vars_to_load = args.variables if variables is None else variables
                s_rho_to_use = args.s_rho if s_rho_slice is None else s_rho_slice
                input_to_use = args.input[0] if len(args.input) == 1 else args.input
                ds = open_dataset(input_to_use, variables=vars_to_load, eta_rho_slice=args.eta_rho, xi_rho_slice=args.xi_rho, s_rho_slice=s_rho_to_use)
                return ds
        
        return load
