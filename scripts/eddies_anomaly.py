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

# assign a number to polarity
EDDY_CYCL_DICT = {
	'none': 0,
	'anticyclonic': 1,
	'cyclonic': 2
}

# assign a number to eddy state
EDDY_STATE_DICT = {
	'unknown': 0,
	'start': 1,
	'stop': 2,
	'continue': 3,
	'fake': 4
}


def init_shard(out_path, eddy_ds, variables, distances, previous_shard, shard_nr):
	"""
	Create a new shard for output
	"""
	print('\n')
	if previous_shard is not None:
		previous_shard.close()
		print('Closed previous shard.')
	if shard_nr is None:
		shard_nr = -1
	shard_nr += 1

	anomaly_path = out_path.replace('.nc', '-{:05d}.nc').format(shard_nr)
	anomaly_ds = nc.Dataset(anomaly_path, 'w')

	anomaly_ds.createDimension("time", None)
	anomaly_ds.createDimension("lfd", None)
	anomaly_ds.createDimension("d2c", distances.shape[0] - 1)

	anomaly_ds.createVariable("time", float, zlib=True, dimensions=('time',), fill_value=np.nan)
	anomaly_ds.variables['time'].units = eddy_ds.time.attrs['units']
	anomaly_ds.variables['time'].calendar = eddy_ds.time.attrs['calendar']
	anomaly_ds.createVariable("doy", int, zlib=True, dimensions=('time',), fill_value=-1)

	## ids
	anomaly_ds.createVariable("tidx", int, zlib=True, dimensions=('time', 'lfd'), fill_value=-3)
	anomaly_ds.createVariable("eidx", int, zlib=True, dimensions=('time', 'lfd'), fill_value=-1)

	anomaly_ds.createVariable("distance_to_coast", float, zlib=True, dimensions=('d2c',), fill_value=np.nan)
	dyolo = (distances[1] - distances[0]) / 2
	anomaly_ds.variables["distance_to_coast"][:] = distances[:-1] + dyolo

	# for each variable x create the following outputs
	# - <x>_mean: average value in eddy instance
	# - <x>_std: standard deviation in eddy instance
	# - <y>_px<z>_sum: sum over all pixels associated to z (z in background, cyclonic, anticyclonic)
	# - <y>_px<z>_mean: mean over all pixels associated to z
	# - <y>_px<z>_std: standard deviation over all pixels associated to z
	# y is on of lg_ (only large eddies), tot_ (all eddies), sm_ (only small eddies)
	# all these outputs are also present with the prefix diff_ which contains the same data but for the deviation from climatology
	for variable in variables:
		for prefix in ['', 'diff_']:
			anomaly_ds.createVariable(prefix+variable+'_mean', float, zlib=True, dimensions=('time', 'lfd'), fill_value=np.nan)
			anomaly_ds.createVariable(prefix+variable+'_std', float, zlib=True, dimensions=('time', 'lfd'), fill_value=np.nan)
			for size_prefix in ['lg_', 'tot_', 'sm_']:
				for type in ['pxbkg', 'pxcycl', 'pxanti']:
					anomaly_ds.createVariable(size_prefix+prefix+variable+'_'+type+'_sum', float, zlib=True, dimensions=('time', 'd2c'), fill_value=np.nan)
					anomaly_ds.createVariable(size_prefix+prefix+variable+'_'+type+'_mean', float, zlib=True, dimensions=('time', 'd2c'), fill_value=np.nan)
					anomaly_ds.createVariable(size_prefix+prefix+variable+'_'+type+'_std', float, zlib=True, dimensions=('time', 'd2c'), fill_value=np.nan)
					
					if prefix == '':
						anomaly_ds.createVariable(size_prefix+prefix+variable+'_'+type+'_area', float, zlib=True, dimensions=('time', 'd2c'), fill_value=np.nan)

	print('Opened new shard at', anomaly_path)

	return anomaly_ds, shard_nr


def id_transform(x):
	# do nothing 
	return x


def run_time_index(
	eddy_ds, ds, climatology, variable, time_index, 
	z_thickness, area_map, distance_map, distances, 
	subdomain, split_area, min_lft=4, days_around=30, reduction='avg', use_abs=False):
	
	# check dates
	date = ds.time.isel(time=time_index).values.item()
	date_eddy = eddy_ds.time.isel(time=time_index).values.item()
	assert abs((date - date_eddy).total_seconds()) < 300
	doy = date.dayofyr - 1

	# get corresponding doys and weights
	doys = get_doys(date, ds, days_around)
	doy_weights = get_triangular_weights(doys)

	# originally variable is U/m³
	# so data is U/m²
	data = ds[variable].isel(time=time_index).values
	diff = data

	# if we have a climatology, calculate the data for this doy
	if climatology is not None:
		doy_weights = doy_weights[:, None, None] if len(climatology.shape) == 3 else doy_weights[:, None, None, None]
		clim = np.nansum(climatology[doys] * doy_weights, axis=0)
		diff = data - clim

	# use absolute values
	if use_abs:
		data = np.abs(data)
		diff = np.abs(diff)
	
	assert reduction in ('avg', 'integrate', 'take')

	# reduce depth by...

	# ... averaging
	if reduction == 'avg':
		data = np.nanmean(data, axis=0)
		diff = np.nanmean(diff, axis=0)
	# ... integrating
	elif reduction == 'integrate':
		data = np.nansum(data * z_thickness[:, None, None], axis=0)
		diff = np.nansum(diff * z_thickness[:, None, None], axis=0)
	# ... taking a specific depth
	else:
		if len(data.shape) == 2:
			pass
		else:
			assert data.shape[0] == 1
			data = data[0]
			diff = diff[0]

	# load ids (time, lfd)
	tidx = eddy_ds.tidx.isel(time=time_index).values
	eidx = eddy_ds.eidx.isel(time=time_index).values
	# load eddy attributes
	eddy_cyc = eddy_ds.cyc.isel(time=time_index).values  # cycl
	eddy_area = eddy_ds.area.isel(time=time_index).values  # area
	eddy_lft = eddy_ds.lifetime.isel(time=time_index).values
	
	# load id maps
	eidx_map = eddy_ds.eidx_map.isel(time=time_index).values
	tidx_map = eddy_ds.tidx_map.isel(time=time_index).values
	cyc_map = np.zeros_like(eidx_map)
	
	# check consistency
	assert eidx_map.shape == area_map.shape, (eidx_map.shape, area_map.shape)
	assert data.shape == eidx_map.shape
	
	# create dictionary with results
	res = {}
	for prefix in ['', 'diff_']:
		# mean value for each eddy instance (U/m²)
		res[prefix+variable+'_mean'] = np.zeros_like(tidx).astype(float) * np.nan
		# standard deviation for each eddy instance (U/m²)
		res[prefix+variable+'_std'] = np.zeros_like(tidx).astype(float) * np.nan
		
		for size_prefix in ['lg_', 'tot_', 'sm_']:
			for type in ['pxbkg', 'pxcycl', 'pxanti']:
				# integrated (mean value * area = U)
				res[size_prefix+prefix+variable+'_'+type+'_sum'] = np.zeros_like(distances[:-1]).astype(float) * np.nan
				# just mean value (U/m²)
				res[size_prefix+prefix+variable+'_'+type+'_mean'] = np.zeros_like(distances[:-1]).astype(float) * np.nan
				res[size_prefix+prefix+variable+'_'+type+'_std'] = np.zeros_like(distances[:-1]).astype(float) * np.nan
				
				# save total area over which integrated (m²), does not depend on prefix
				if prefix == '':
					res[size_prefix+prefix+variable+'_'+type+'_area'] = np.zeros_like(distances[:-1]).astype(float) * np.nan

	
	# DEBUG
	control_sums = []  

	# create maps that can be used for calculating px<z> stuff
	data_cp_tot = data.copy()
	data_cp_lg = data.copy()
	data_cp_sm = data.copy()

	for i, eddy_idx in enumerate(eidx):  # loop eddies
		# create a bool map for the eddy
		bool_map = eidx_map == eddy_idx
		if np.count_nonzero(bool_map) > 0:
			# check that track is unique/track matches the eddy bool map
			shown_tidx = np.unique(tidx_map[bool_map])
			assert shown_tidx.shape[0] == 1, shown_tidx
			# use data only if above min lifetime
			if eddy_lft[i] >= min_lft:
				data_c = data_cp_lg if eddy_area[i] >= split_area else data_cp_sm  # choose correct map
				data_cp_tot[bool_map] = np.nan
				data_c[bool_map] = np.nan
				# debug
				control_sums.append(np.nansum(data[bool_map] * area_map[bool_map]))  # DEBUG
			
			# save mean (weighted average with weights the area of the single pixels)
			lolmap = ~np.isnan(data[bool_map])
			res[variable+'_mean'][i] = 0 if np.nansum(area_map[bool_map][lolmap]) == 0 else np.average(data[bool_map][lolmap], weights=area_map[bool_map][lolmap])
			res['diff_'+variable+'_mean'][i] = 0 if np.nansum(area_map[bool_map][lolmap]) == 0 else np.average(diff[bool_map][lolmap], weights=area_map[bool_map][lolmap])
			# save std
			res[variable+'_std'][i] = np.nanstd(data[bool_map])
			res['diff_'+variable+'_std'][i] = np.nanstd(diff[bool_map])
			
			# save cycl
			cyc_map[bool_map] = eddy_cyc[i]

	# DEBUG
	assert np.allclose(np.nansum(control_sums) + np.nansum(data_cp_tot * area_map), np.nansum(data * area_map)), (np.nansum(control_sums), np.nansum(data_cp_tot * area_map), np.nansum(data * area_map))

	for data_cp, size_prefix in [(data_cp_tot, 'tot_'), (data_cp_lg, 'lg_'), (data_cp_sm, 'sm_')]:
		# loop distance bins
		for i in range(distances.shape[0] - 1):
			# create boolean mask for distances
			dist_bool_map = np.logical_and(distance_map >= distances[i], distance_map < distances[i+1])
			# only consider subdomain
			dist_bool_map = np.logical_and(dist_bool_map, subdomain)
			# save sum of full domain (bkg + eddies)
			
			px_bool_maps = {
				'pxbkg': np.logical_and(dist_bool_map, ~np.isnan(data_cp)),  # remove eddies (nan in data_cp) from bool mask
				'pxcycl': np.logical_and(dist_bool_map, np.logical_and(cyc_map == 2, np.isnan(data_cp))),
				'pxanti': np.logical_and(dist_bool_map, np.logical_and(cyc_map == 1, np.isnan(data_cp)))
			}	

			for name, name_px_boolmap in px_bool_maps.items():
				for prefix, k in [('', data), ('diff_', diff)]:
					# save sum of bkg (U/m² * km² = 1e6*U)
					res[size_prefix+prefix+variable+'_'+name+'_sum'][i] = np.nansum(k[name_px_boolmap] * area_map[name_px_boolmap])
					# save mean of bkg (U/m²)
					res[size_prefix+prefix+variable+'_'+name+'_mean'][i] = np.nanmean(k[name_px_boolmap])
					res[size_prefix+prefix+variable+'_'+name+'_std'][i] = np.nanstd(k[name_px_boolmap])
			
				# save area of bkg (independent of prefix)
				res[size_prefix+variable+'_'+name+'_area'][i] = np.nansum(area_map[name_px_boolmap])

	res['tidx'] = tidx
	res['eidx'] = eidx

	return res


def write_to_shard(anomaly_ds, in_shard_idx, date, variable, res):
	# conversion function for date objects
	conv_fn = partial(date2num, units=anomaly_ds.variables['time'].units, calendar=anomaly_ds.variables['time'].calendar)
	# write data to output
	anomaly_ds.variables['time'][in_shard_idx] = conv_fn(date)
	anomaly_ds.variables['doy'][in_shard_idx] = date.dayofyr - 1
	anomaly_ds.variables['tidx'][in_shard_idx, :res['tidx'].shape[0]] = res['tidx'].astype(int)
	anomaly_ds.variables['eidx'][in_shard_idx, :res['eidx'].shape[0]] = res['eidx'].astype(int)

	for res_key in res:
		if res_key == 'tidx' or res_key == 'eidx':
			continue
		anomaly_ds.variables[res_key][in_shard_idx, :res[res_key].shape[0]] = res[res_key]


if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--input", type=str, help="Input dataset", required=True)
	parser.add_argument("-dsp", "--dataset-path", type=str, help="Input dataset", default='z/z_d_*.nc')
	
	parser.add_argument("-v", "--variable", type=str, help="Variable", required=True)
	parser.add_argument("-o", "--output", type=str, help="Output path", required=True)
	parser.add_argument("-n", "--num-distances", type=int, help="Number of ranges from 0 to 1000", default=51)
	parser.add_argument("--days-around", type=int, help="Number of days around", default=30)
	parser.add_argument("--depth", type=parse_slice, help="Depth slice", default=slice(None, None))
	parser.add_argument("--reduction", type=str, choices=('avg', 'integrate', 'take'), help="Reduction method", required=True)
	parser.add_argument("--abs", action='store_true', help="Use absolute value")
	
	args = parser.parse_args()
	
	# the folder requires a specific structure
	base_path = '/work/maxsimon/data/'+args.input

	# load data
	ds = open_dataset(os.path.join(base_path, args.dataset_path), ['time', args.variable])
	# load grid
	grid = xr.open_dataset(os.path.join(base_path, 'grid.nc'))
	area_map = get_area_map(grid)
	z_data = xr.open_dataset(os.path.join(base_path, 'z/z_levels.nc'))
	z_thickness = z_data['thickness_z'].values
	print('Depths:', z_data['z_level'].values[args.depth])
	grid_data = np.load(os.path.join(base_path, 'grid.npz'))
	distance_map = grid_data['distance_map']
	subdomain = grid_data['gruber_mask']

	distances = np.linspace(0, 1000, args.num_distances).astype(float)
	print('Distances:', distances)

	print('Use abs:', args.abs)

	# define which eddies to put into sm_ and lg_ category
	split_area = np.pi * (30.0 ** 2)

	# load eddies
	eddy_ds = open_dataset(os.path.join(base_path, 'ssh/eddies-00000.nc'))  #xr.open_dataset(args.eddies)

	# create output
	time_idxs = list(range(ds.dims['time']))
	anomaly_ds, _ = init_shard(args.output, eddy_ds, [args.variable], distances, None, None)

	# load climatology
	climatology = None
	if args.days_around > 0:
		climatology = xr.open_dataset(os.path.join(base_path, 'climatologies/smooth/clim-0deg-'+args.variable+'_b.nc'))[args.variable+'_b']
		if len(climatology.shape) == 4:
			climatology = climatology.isel(depth=args.depth)
			ds = ds.isel(depth=args.depth)
		else:
			assert args.reduction == 'take'

		climatology = climatology.values
		print('Climatology loaded')

	else:
		print('Do not load any climatolog')

	# loop time
	for i, time_idx in enumerate(time_idxs):
		# get time object
		date = ds.time.isel(time=time_idx).values.item()
		print('Processing', date, end='\r')
		# process data
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			res = run_time_index(
				eddy_ds, 
				ds, 
				climatology, 
				args.variable, 
				time_idx, 
				z_thickness[args.depth], 
				area_map, 
				distance_map, 
				distances, 
				subdomain,
				split_area=split_area,
				days_around=args.days_around,
				reduction=args.reduction,
				use_abs=args.abs
			)
		# output data
		write_to_shard(anomaly_ds, i, date, args.variable, res)

		if i == len(time_idxs) - 1:
			anomaly_ds.close()
			print('\nClosed last shard.')



	
