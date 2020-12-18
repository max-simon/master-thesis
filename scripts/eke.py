#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


import sys
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')

import os
import numpy as np
import xarray as xr
import netCDF4 as nc

from datetime import timedelta as tdelta
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from romstools.dataset import dataset_from_args
from romstools.utils import parse_slice, parse_datetime_string, np_rolling_mean, check_output_path, get_depth_dim, get_doys, get_triangular_weights

from functools import partial
from cftime import date2num

import warnings
import progressbar


def eke_at_depth(ds, climatology, depth_idx, time_slice=slice(None, None), days_around=30):
	"""
	Calculate EKE and associated data at a given depth index. This function is efficient, when the whole domain x time fits into memory.
	"""

	# get times
	times = np.squeeze(ds['time'].values)
	# get depth key
	depth_key = get_depth_dim(ds)

	# load raw data
	raw_u = ds['u'].isel(**{depth_key: depth_idx, 'time': time_slice}).values
	print('Raw u loaded')
	raw_v = ds['v'].isel(**{depth_key: depth_idx, 'time': time_slice}).values
	print('Raw v loaded')
	# load climatological data
	clim_u = climatology['u_b'].isel(**{depth_key: depth_idx}).values
	print('Clim u loaded')
	clim_v = climatology['v_b'].isel(**{depth_key: depth_idx}).values
	print('Clim v loaded')

	# add memory for prime values = value - climatology
	prime_u = raw_u.copy()
	prime_v = raw_v.copy()

	# add memory for results
	eke_b = np.zeros_like(clim_u)	
	num_items = np.zeros(clim_u.shape[0])

	# loop times
	for t_idx, t_obj in progressbar.progressbar(enumerate(times)):
		# get doys and weights
		doy = t_obj.dayofyr - 1
		doys = get_doys(t_obj, ds, 30)
		doys_weights = get_triangular_weights(doys)
		# calculate the climatological value
		c_u = np.nansum(clim_u[doys] * doys_weights[:, None, None], axis=0)
		c_v = np.nansum(clim_v[doys] * doys_weights[:, None, None], axis=0)
		# calculate prime values
		prime_u[t_idx] -= c_u
		prime_v[t_idx] -= c_v
		num_items[doy] += 1
		# add eke-climatological
		eke_b[doy] += 0.5*((prime_u[t_idx]**2) + (prime_v[t_idx]**2))
	# normalize eke-climatological
	eke_b /= num_items[:, None, None]

	print('Calculated primes')
	
	# calculate eke
	eke = 0.5*((prime_u**2) + (prime_v**2))
	# calculate ke
	ke = 0.5*(raw_u**2 + raw_v**2)

	return eke, eke_b, prime_u, prime_v, ke


def create_eke_output(output_path, dataset):
	"""
	Create a writeable netCDF4 object with the eke structure
	"""

	# check if output exists
	res = check_output_path(output_path)
	if res is not None:
		return res

	# get depth key
	depth_key = get_depth_dim(dataset)

	# get all dimensions
	dims = {
		'eta_rho': dataset['u'].shape[2],
		'xi_rho': dataset['u'].shape[3],
		'time': dataset['u'].shape[0],
		'doy': 365
	}

	# set up output dataset
	ds_out = nc.Dataset(output_path, 'w')  # pylint: disable=no-member
	ds_out.setncatts(dataset.attrs)

	# create dimensions
	chunksizes = {}
	for dim in dims:
		ds_out.createDimension(dim, dims[dim])
		chunksizes[dim] = dims[dim]
	chunksizes['time'] = 30
	chunksizes['doy'] = 30

	print('Chunking:', chunksizes)

	# create time variable
	ds_out.createVariable('time', float, zlib=True, dimensions=('time', ), fill_value=np.nan)
	ds_out.variables['time'].units = dataset.time.attrs['units']
	ds_out.variables['time'].calendar = dataset.time.attrs['calendar']
	# save time values
	conv_fn = partial(date2num, units=dataset.time.attrs['units'], calendar=dataset.time.attrs['calendar'])
	ds_out['time'][:] = np.array([conv_fn(t_obj) for t_obj in dataset.time.values])

	# create variables with correct chunking
	for variable in ['eke', 'eke_b', 'u_p', 'v_p', 'ke']:
		dimensions = ['time', 'eta_rho', 'xi_rho']
		if variable == 'eke_b':
			dimensions[0] = 'doy'
		variable_chunking = tuple([chunksizes[dim] for dim in dimensions])
		ds_out.createVariable(variable, float, zlib=True, dimensions=tuple(dimensions), fill_value=np.nan, chunksizes=variable_chunking)
		
	return ds_out


if __name__ == "__main__":

	import argparse
	import progressbar

	# Define arguments
	parser = argparse.ArgumentParser()
	get_dataset = dataset_from_args(parser)
	parser.add_argument("-o", "--output", type=str, help="Output path for eke file", required=True)
	parser.add_argument("-c", "--climatology", type=str, help="Climatology")
	parser.add_argument("--days-around", type=int, help="Days around", default=30)
	parser.add_argument("--depth", type=int, help="Depth index", default=0)

	args = parser.parse_args()

	# get initial dataset
	dsf = get_dataset(args)

	climatology = xr.open_dataset(args.climatology)
	
	# get output
	ds_out = create_eke_output(args.output, dsf)

	# do calculation
	print('Start calculation')
	eke, eke_b, prime_u, prime_v, ke = eke_at_depth(dsf, climatology, args.depth)
	print('Calculation done')

	# save output
	ds_out['eke'][:] = eke
	ds_out['eke_b'][:] = eke_b
	ds_out['u_p'][:] = prime_u
	ds_out['v_p'][:] = prime_v
	
	dsf.close()
	ds_out.close()
