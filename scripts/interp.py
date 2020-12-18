#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


import sys
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')

import xesmf as xe
import numpy as np
import xarray as xr

from romstools.dataset import open_glob_dataset
from romstools.utils import parse_slice
import subprocess


def get_dataset_for_variable(path_data, path_grid, variable):
	"""
	Open a dataset with a single variable, get the correct grid name and modify corresponding grid
	such that it has lon, lat
	"""
	ds = open_glob_dataset(path_data, keep_vars=[variable, 'time'])
	grid = xr.open_dataset(path_grid)

	grid_name = 'rho'

	# get the correct grid_name
	for dim in ds[variable].dims:
		postfix = dim.replace('lon_', '').replace(
			'lat_', '').replace('eta_', '').replace('xi_', '')
		if postfix != dim and postfix != grid_name and postfix != 'rho':
			grid_name = postfix
	
	# rename dimensions to match the grid_name
	for dim in ds[variable].dims:
		postfix = dim.replace('lon_', '').replace(
			'lat_', '').replace('eta_', '').replace('xi_', '')
		if postfix != dim and postfix != grid_name:
			new_dim_name = dim.replace(postfix, grid_name)
			ds = ds.rename_dims(**{dim: new_dim_name})

	ds = ds.assign_coords(
		lon=grid['lon_'+grid_name], lat=grid['lat_'+grid_name])

	return ds, grid_name


def get_interpolator(ds, grid_out, grid_name, method):
	"""
	Create an xesmf interpolator
	"""
	ds_out = grid_out.rename(**{'lon_'+grid_name: 'lon', 'lat_'+grid_name: 'lat'})
	regridder = xe.Regridder(ds, ds_out, method, reuse_weights=True)
	return regridder



if __name__ == "__main__":

	import argparse

	# class to parse a key=value array into a dictionary
	class StoreDictKeyPair(argparse.Action):
		def __init__(self, option_strings, dest, nargs=None, **kwargs):
			self._nargs = nargs
			super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

		def __call__(self, parser, namespace, values, option_string=None):
			my_dict = {}
			for kv in values:
				k, v = kv.split("=")
				my_dict[k] = v
			setattr(namespace, self.dest, my_dict)

	# Define arguments
	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--input", type=str, nargs="+",
						help="Input file", required=True)
	parser.add_argument("-o", "--output", type=str,
						help="Output path", required=True)
	parser.add_argument("-gi", "--grid-input", type=str,
						help="Grid file for input file", required=True)
	parser.add_argument("-go", "--grid-output", type=str,
						help="Grid file for output")
	parser.add_argument("--target-grid-name", nargs="+", action=StoreDictKeyPair,
						help="Target prefix for interpolation", default={})
	parser.add_argument("--method", choices=['bilinear', 'conservative', 'nearest_s2d',
						'nearest_d2s', 'patch'], help="Interpolation method", default='bilinear')
	parser.add_argument("--variables", type=str, nargs="+",
						help="Choose variables to load", default=[])
	parser.add_argument("-d", "--depth-slice", type=parse_slice, help="Slice for depth")
	parser.add_argument("-t", "--time-slice", type=parse_slice, help="Slice for time")

	args = parser.parse_args()

	input_to_use = args.input[0] if len(args.input) == 1 else args.input

	# use grid-input for grid-output if not given
	if args.grid_output is None:
		args.grid_output = args.grid_input
	print('Use target grid', args.grid_output)

	if len(args.variables) == 0:
		# load dataset
		ds = open_glob_dataset(input_to_use)
		args.variables = list(ds.variables)

	# loop variables to process
	for variable in args.variables:
		print('Processing variable', variable)

		ds, grid_name = get_dataset_for_variable(input_to_use, args.grid_input, variable)
		# slice on depth
		if args.depth_slice is not None:
			print('\tUse only depths', args.depth_slice)
			ds = ds.isel(depth=args.depth_slice)
		# slice on time
		if args.time_slice is not None:
			print('\tUse only times', args.time_slice)
			ds = ds.isel(time=args.time_slice)

		# you need to create an output dataset for xesmf
		ds_out = xr.open_dataset(args.grid_output)

		# change grid if set for variable
		if variable in args.target_grid_name:
			print('\tInstead of grid {:s} use grid {:s}'.format(grid_name, args.target_grid_name[variable]))
			grid_name = args.target_grid_name[variable]
		
		# create interpolator
		regridder = get_interpolator(ds, ds_out, grid_name, args.method)

		# interpolate
		var_out = regridder(ds[variable])
		# save to file
		output_path = args.output.replace('.nc', '-{:s}.nc'.format(variable))
		print('\tSave to {:s}'.format(output_path))
		var_out.to_netcdf(output_path)
		ds.close()
