#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


import sys
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')

import numpy as np
import xarray as xr
import scipy.io as sio
from collections import defaultdict
import time
import os
from romstools.dataset import open_dataset
from romstools.utils import get_area_map, get_doys, get_triangular_weights, parse_slice
import netCDF4 as nc
from cftime import date2num
from functools import partial
from geopy.distance import geodesic
import warnings

from matplotlib import pyplot as plt


def init_shard(out_path, ds, variables, distances):
	"""
	Create a new shard for output
	"""

	anomaly_ds = nc.Dataset(out_path, 'w')

	# create time
	anomaly_ds.createDimension("time", None)
	anomaly_ds.createVariable("time", float, zlib=True, dimensions=('time',), fill_value=np.nan)
	anomaly_ds.variables['time'].units = ds.time.attrs['units']
	anomaly_ds.variables['time'].calendar = ds.time.attrs['calendar']
	
	# create spatial dimensions
	anomaly_ds.createDimension("eta_rho", ds.dims['eta_rho'])
	anomaly_ds.createDimension("xi_rho", ds.dims['xi_rho'])
	anomaly_ds.createDimension("d2c", distances.shape[0] - 1)

	dimensions_map = ('time', 'eta_rho', 'xi_rho')
	variable_chunking_map = (30, ds.dims['eta_rho'], ds.dims['xi_rho'])

	dimensions_d2c = ('time', 'd2c')
	variable_chunking_d2c = (90, distances.shape[0] - 1)

	# create distance to coast data
	anomaly_ds.createVariable("distance_to_coast", float, zlib=True, dimensions=('d2c',), fill_value=np.nan)
	dyolo = (distances[1] - distances[0]) / 2
	anomaly_ds.variables["distance_to_coast"][:] = distances[:-1] + dyolo

	for prefix in ['tot_', 'lg_', 'sm_']:  # see eddies_anomaly.py for description.
		for var in variables:
			# create full map
			anomaly_ds.createVariable(prefix+var+"_map", float, zlib=True, dimensions=dimensions_map, fill_value=np.nan, chunksizes=variable_chunking_map)
			# create hovmöller data (for cyclones and anticyclones)
			anomaly_ds.createVariable(prefix+var+"_d2c_cycl", float, zlib=True, dimensions=dimensions_d2c, fill_value=np.nan, chunksizes=variable_chunking_d2c)
			anomaly_ds.createVariable(prefix+var+"_d2c_anti", float, zlib=True, dimensions=dimensions_d2c, fill_value=np.nan, chunksizes=variable_chunking_d2c)
			for key in ds[var].attrs:
				setattr(anomaly_ds.variables[prefix+var+'_map'], key, ds[var].attrs[key])
				setattr(anomaly_ds.variables[prefix+var+'_d2c_cycl'], key, ds[var].attrs[key])
				setattr(anomaly_ds.variables[prefix+var+'_d2c_anti'], key, ds[var].attrs[key])

	return anomaly_ds


def build_map(eddy_ds, variables, distance_map, subdomain, distances, min_lft, split_area):
	# load map data
	eidx_map = eddy_ds.eidx_map.values
	cyc_map_lg = np.zeros_like(eidx_map).astype(int)
	cyc_map_sm = np.zeros_like(eidx_map).astype(int)
	cyc_map_tot = np.zeros_like(eidx_map).astype(int)

	# load eddy data
	eidxs = eddy_ds.eidx.values
	data = {
		var: eddy_ds[var].values for var in variables
	}

	# get polarities
	cyc_data = eddy_ds.cyc.values
	# get lifetimes
	lft_data = eddy_ds.lifetime.values.astype(int)
	# get area of eddies
	area_data = eddy_ds.area.values
	assert np.max(lft_data > 2)

	def get_result_dict():
		# set up result dict
		res = dict(**{
			var+'_d2c_cycl': np.zeros(distances.shape[0] - 1).astype(float) for var in variables
		}, **{
			var+'_d2c_anti': np.zeros(distances.shape[0] - 1).astype(float) for var in variables
		}, **{
			var+'_map': np.zeros_like(eidx_map).astype(float) for var in variables
		})
		return res

	# create different result dicts
	res_lg = get_result_dict()
	res_sm = get_result_dict()
	res_tot = get_result_dict()

	num_lg = 0
	num_sm = 0

	for i, eix in enumerate(eidxs):  # loop all eddies
		if eix > 0:
			m = eidx_map == eix
			# drop eddies that live too short
			if lft_data[i] >= min_lft:
				# choose in which one to save additionally
				res_s = None
				cyc_map_s = None
				# choose correct result dict (lg_ or sm_)
				if area_data[i] >= split_area:
					res_s = res_lg
					cyc_map_s = cyc_map_lg
					num_lg += 1
				else:
					res_s = res_sm
					cyc_map_s = cyc_map_sm
					num_sm += 1
				
				# save in total and correct res
				for var in variables:
					res_tot[var+'_map'][m] = data[var][i]
					res_s[var+'_map'][m] = data[var][i]
				
				# save in total and correct cyc_map
				cyc_map_tot[m] = cyc_data[i]
				cyc_map_s[m] = cyc_data[i]
	
	# after creating the maps, we can easily calculate the hovmöller data
	for res, cyc_map in [(res_lg, cyc_map_lg), (res_sm, cyc_map_sm), (res_tot, cyc_map_tot)]:
		for i in range(distances.shape[0] - 1):
			# create boolean mask for distances
			dist_bool_map = np.logical_and(distance_map >= distances[i], distance_map < distances[i+1])
			# only consider subdomain
			dist_bool_map = np.logical_and(dist_bool_map, subdomain)
			# split for direction
			bool_map_cycl = np.logical_and(dist_bool_map, cyc_map == 2)
			bool_map_anti = np.logical_and(dist_bool_map, cyc_map == 1)
			for var in variables:
				res[var+'_d2c_cycl'][i] = np.nanmean(res[var+'_map'][bool_map_cycl])
				res[var+'_d2c_anti'][i] = np.nanmean(res[var+'_map'][bool_map_anti])

	return res_tot, res_lg, res_sm, num_lg/num_sm


def write_to_shard(anomaly_ds, in_shard_idx, date, res, prefix):
	# conversion function for date objects
	conv_fn = partial(date2num, units=anomaly_ds.variables['time'].units, calendar=anomaly_ds.variables['time'].calendar)
	# write data to output
	anomaly_ds.variables['time'][in_shard_idx] = conv_fn(date)
	for var in res:
		anomaly_ds.variables[prefix+var][in_shard_idx] = res[var]


def get_intensity(eddies):
    # calculate radius from eddy area, in km
    radius = np.sqrt(eddies.area/np.pi)
    # return cm/km
    return eddies.amplitude/radius


if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser()

	# setup argument parser
	parser.add_argument("-i", "--input", type=str, help="Input dataset", required=True)
	parser.add_argument("-o", "--output", type=str, help="Output path", required=True)
	parser.add_argument("-n", "--num-distances", type=int, help="Number of ranges from 0 to 1000", default=51)
	parser.add_argument('-lft', '--min-lifetime', type=int, help="Minimum lifetime of eddies to consider", default=4)
	parser.add_argument('-spr', '--split-radius', type=float, help="Radius to split lg and sm", default=30.0)
	args = parser.parse_args()

	base_path = '/nfs/kryo/work/maxsimon/data/'+args.input
	vars = ['intensity', 'cyc', 'amplitude']  # define which variables to use

	# load data
	ds = open_dataset(os.path.join(base_path, 'ssh/eddies-00000.nc'))
	ds = ds.assign(intensity=get_intensity(ds))
	print(ds)

	# load grid data
	grid = xr.open_dataset(os.path.join(base_path, 'grid.nc'))
	area_map = get_area_map(grid)
	grid_data = np.load(os.path.join(base_path, 'grid.npz'))
	distance_map = grid_data['distance_map']
	subdomain = grid_data['subdomain']

	# create distance array for hovmöller data
	distances = np.linspace(0, 1000, args.num_distances).astype(float)
	
	# create output shard
	out_ds = init_shard(args.output, ds, vars, distances)
	# define which eddies to put into sm_ and lg_ category
	split_area = np.pi * (args.split_radius ** 2)

	# loop times
	for time_index, t_obj in enumerate(ds.time.values):
		# calculate maps and hovmöller data
		res_tot, res_lg, res_sm, ratio = build_map(ds.isel(time=time_index), vars, distance_map, subdomain, distances, args.min_lifetime, split_area)
		# output data
		for res, prefix in [(res_lg, 'lg_'), (res_sm, 'sm_'), (res_tot, 'tot_')]:
			write_to_shard(out_ds, time_index, t_obj, res, prefix)
		print('Processing', t_obj, 'Ratio is', ratio, end="\r")
