#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


import sys
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')

import xarray as xr
import os
import netCDF4 as nc
import numpy as np
from time import time
import warnings
from romstools.dataset import dataset_from_args
from romstools.utils import check_output_path, get_depth_dim, get_num_days


def get_doy_mask(doy, days_around, dataset):
	"""
	Create a boolean mask for a dataset where all entries are True if they are in the range of days_around around
	the day-of-the-year (doy).
	"""
	if days_around == 0:
		return dataset.doy == doy
	max_days = get_num_days(dataset)
	# at edge when values of december are needed for january (or otherway around)
	at_edge = False
	# get doy_min, fix negative values
	doy_min = doy - days_around
	if doy_min < 0:
		doy_min = max_days + doy_min
		at_edge = True
	# get doy_max, fix too large values
	doy_max = doy + days_around
	if doy_max >= max_days:
		doy_max = doy_max - max_days
		at_edge = True
	# mask is logical_or if at edge and logical_and if not
	mask = xr.ufuncs.logical_or(dataset.doy >= doy_min, dataset.doy <= doy_max) if at_edge else xr.ufuncs.logical_and(dataset.doy >= doy_min, dataset.doy <= doy_max)
	return mask


def calculate_climatology_at_depth(ds, variable, depth_idx, days_around=0, with_std=True):
	"""
	Calculate the climatology of a dataset ds for a variable at depth depth_idx. Use days_around for smoothing and write 
	results to ds_out.
	"""

	# load raw data into memory
	depth_key = get_depth_dim(ds)
	assert (depth_key is not None) or depth_idx is None
 
	data_t = ds[variable].isel(**{depth_key: depth_idx}).values if depth_key is not None else ds[variable].values
	# get the number of days (number of doys)
	num_days = get_num_days(ds)
	new_shape = tuple([num_days] + list(data_t.shape[1:]))

	# set up data arrays for climatology, its standard deviation and the  number of items
	data_clim = np.empty(new_shape, dtype=float)
	data_clim_std = None if not with_std else np.empty(new_shape, dtype=float)
	data_clim_num = np.empty((num_days, ), dtype=int)

	# loop doys
	for doy in range(num_days):
		# create boolean mask
		mask = get_doy_mask(doy, days_around, ds)
		# get data
		data = data_t[mask]
		assert data.shape[0] > 0, (data.shape, np.count_nonzero(mask))

		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", "Degrees of freedom <= 0 for slice")
			warnings.filterwarnings("ignore", "Mean of empty slice")
			# calculate data
			data_clim[doy] = np.nanmean(data, axis=0)
			if with_std:
				data_clim_std[doy] = np.nanstd(data, axis=0)
			data_clim_num[doy] = data.shape[0]
	
	return data_clim, data_clim_std, data_clim_num
	

def write_climatology_at_depth(ds_out, variable, depth_idx, data_clim, data_clim_std, data_clim_num, with_std=True):
	if depth_idx is None:
		# write data to file
		ds_out[variable+'_b'][:] = data_clim
		if with_std:
			ds_out[variable+'_std'][:] = data_clim_std
		ds_out[variable+'_num'][:] = data_clim_num
	else:
		# write data to file
		ds_out[variable+'_b'][:, depth_idx] = data_clim
		if with_std:
			ds_out[variable+'_std'][:, depth_idx] = data_clim_std
		ds_out[variable+'_num'][:] = data_clim_num


def create_climatology_output(output_path, dataset, variables, days_around, with_std=False):
	"""
	Create a writeable netCDF4 object with the climatology structure
	"""

	# check if output exists
	res = check_output_path(output_path)
	if res is not None:
		# check if days_around matches
		ds_days_around = res.getncattr('days_around')
		if ds_days_around != days_around:
			raise RuntimeError('Found different days_around attribute in file:', days_around)
		return res

	# get number of days per year
	max_days = get_num_days(dataset)
	# get all dimensions
	dims = {}
	for variable in variables:
		for i, dim in enumerate(dataset[variable].dims):
			if dim != "time" and dim != "doy":
				dims[dim] = dataset[variable].shape[i]
	
	# get depth key
	depth_key = get_depth_dim(dataset)

	# set up output dataset
	ds_out = nc.Dataset(output_path, 'w')  # pylint: disable=no-member
	ds_out.setncatts(dataset.attrs)
	
	# save the number of days_around used for this
	ds_out.setncattr('days_around', days_around)

	chunksizes = {
		'doy': 1
	}

	# create dimensions
	ds_out.createDimension("doy", max_days)
	for dim in dims:
		ds_out.createDimension(dim, dims[dim])
		chunksizes[dim] = dims[dim] if dim != depth_key else min(10, dims[depth_key])

	print('Chunking:', chunksizes)
	
	for variable in variables:
		dimensions = ['doy'] + list([dim for dim in dataset[variable].dims if dim != "time" and dim != "doy"])
		variable_chunking = tuple([chunksizes[dim] for dim in dimensions])
		# only variable or also with standard deviation
		variable_names = [variable + '_b'] if not with_std else [variable + '_b', variable + '_std']
		# create all the variables
		for variable_name in variable_names:
			ds_out.createVariable(variable_name, float, zlib=True, dimensions=tuple(dimensions), fill_value=np.nan, chunksizes=variable_chunking)
			for key in dataset[variable].attrs:
				setattr(ds_out.variables[variable_name], key, dataset[variable].attrs[key])
		ds_out.createVariable(variable+'_num', int, zlib=True, dimensions=("doy",), fill_value=0)

	return ds_out



if __name__ == "__main__":

	import argparse
	import progressbar

	# Define arguments
	parser = argparse.ArgumentParser()
	get_dataset = dataset_from_args(parser)

	parser.add_argument("-o", "--output", type=str, help="Output path for smoothed climatology", required=True)
	parser.add_argument("--days-around", type=int, help="Number of days around doy for smoothed climatology", default=0)

	args = parser.parse_args()

	# get initial dataset
	dsf = get_dataset(args)
	
	depth_key = get_depth_dim(dsf)
	depth_idxs = [None] if depth_key is None else list(range(dsf.dims[depth_key]))

	print(depth_idxs)

	# get output netCDF file
	out = create_climatology_output(args.output, dsf, args.variables, args.days_around, with_std=False)

	# loop vars
	for var in args.variables:
		print('Start calculation of', var)
		for depth_idx in progressbar.progressbar(depth_idxs):
			# reopen dataset
			dsf = get_dataset(args)
			# calculate data
			data_clim, data_clim_std, data_clim_num = calculate_climatology_at_depth(dsf, var, depth_idx, args.days_around, with_std=False)
			# write data
			write_climatology_at_depth(out, var, depth_idx, data_clim, data_clim_std, data_clim_num, with_std=False)
			# close dataset to drop in-memory data
			dsf.close()
	out.close()

