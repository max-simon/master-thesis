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
from romstools.utils import check_output_path, get_depth_dim, get_num_days, get_doys, get_triangular_weights
from romstools.dataset import dataset_from_args, open_dataset
from cftime import date2num
from functools import partial

from concurrent.futures import ProcessPoolExecutor


def multithreading(func, args, workers):
	"""
	Execute a list of functions with a list of arguments in parallel.
	"""
	with ProcessPoolExecutor(workers) as ex:
		res = ex.map(func, args)
	return list(res)


def create_output(output_path, dataset, days_around):
	"""
	Create a writeable netCDF4 object with the climatology structure
	"""

	# check if output exists
	res = check_output_path(output_path)
	if res is not None:
		return res

	# copy all dims
	dims = {}
	for variable in dataset.variables:
		for i, dim in enumerate(dataset[variable].dims):
			dims[dim] = dataset[variable].shape[i]

	# set up output dataset
	ds_out = nc.Dataset(output_path, 'w')  # pylint: disable=no-member
	ds_out.setncatts(dataset.attrs)

	# save the used days_around
	ds_out.setncattr('days_around', days_around)

	# create all dimensions
	for dim in dims:
		ds_out.createDimension(dim, dims[dim])

	# copy variables, but append a _p (except for time)
	for variable in dataset.variables:
		dimensions = list([dim for dim in dataset[variable].dims])
		new_var_name = variable + '_p' if variable != 'time' else 'time'
		ds_out.createVariable(new_var_name, float, zlib=True, dimensions=tuple(dimensions), fill_value=np.nan)
		for k in dataset[variable].attrs:
			setattr(ds_out.variables[new_var_name], k, dataset[variable].attrs[k])

	return ds_out


def do(a):
	# unroll a
	t_idx, ds_data, days_around, clim_data = a
	# get time object and doy
	t_obj = ds.time.isel(time=t_idx).values.item()
	# get doy and calculate the doys affected
	doy = t_obj.dayofyr - 1
	doys = get_doys_around_doy(doy, 365, days_around)
	doys_weights = get_triangular_weights(doys)
	# substract the climatology from each grid point
	ds_data -= np.nansum(clim_data[doys] * doys_weights[:, None, None, None], axis=0)
	return ds_data


if __name__ == "__main__":

	import argparse
	import progressbar

	# Define arguments
	parser = argparse.ArgumentParser()
	get_dataset = dataset_from_args(parser)

	parser.add_argument("-o", "--output", type=str, help="Output path", required=True)
	parser.add_argument("-c", "--climatology", type=str, help="Climatology file", required=True)
	parser.add_argument("--days-around", type=int, help="Number of days around doy for smoothed climatology", default=15)
	parser.add_argument("--num-threads", type=int, help="Number of threads to use", default=10)

	args = parser.parse_args()

	# get initial dataset
	dsf = get_dataset(args)
	conv_fn = partial(date2num, units=dsf.time.attrs['units'], calendar=dsf.time.attrs['calendar'])
	
	# get variables as a list
	variables = list([var for var in dsf.variables])

	# create output
	output = create_output(args.output, dsf, args.days_around)

	# open climatology
	clim = open_dataset(args.climatology, eta_rho_slice=args.eta_rho, xi_rho_slice=args.xi_rho, s_rho_slice=args.s_rho)

	# close initial dataset
	dsf.close()

	# define the number of threads to use
	num_threads = args.num_threads

	# loop variables
	for var in variables:
		# do not process time
		if var == 'time' or var == 'doy':
			continue

		# open dataset
		ds = get_dataset(args, [var])
		# load climatology (this takes some time)
		clim_data = clim[var+'_b'].values
		print('Climatology loaded', clim_data.shape)

		# loop times
		times = ds.time.values
		for t_idx in progressbar.progressbar(list(range(0, len(times), num_threads))):
			# get data for that time slice
			ds_data = ds[var].isel(time=slice(t_idx, t_idx+num_threads)).values
			# set up arguments for calculation (see do function for order)
			fixed_args = [args.days_around, clim_data]
			pool_args = [
				[t_idx + thread_idx, ds_data[thread_idx]] + fixed_args for thread_idx in range(num_threads)
			]
			# submit jobs
			res = multithreading(do, pool_args, num_threads)
			# push to output
			output[var+'_p'][t_idx:t_idx+num_threads, :] = np.array(res)
			
		ds.close()
	output.close()
