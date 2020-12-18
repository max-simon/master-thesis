#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


import sys
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')
import numpy as np
import xarray as xr
from collections import defaultdict
from cftime import date2num
from romstools.dataset import dataset_from_args
from functools import partial
from scipy.interpolate import griddata
import warnings
from romstools.utils import parse_slice, get_num_days, get_doys, get_triangular_weights
from romstools.slice import slice_on_rho_grid
from romstools.dataset import open_dataset

######################################################################
#
#  The script requires a major refactoring. Actually it would
#  be smarter to calculate only the slices for a given filtering,
#  saving them into a source file and then calculate anomalies
#  based on these slices. In addition, the drop_nan option should be
#  dropped, because some data can produce nans, some not. This leads
#  to a misalignment of the different shards!
#
######################################################################

# Parameters for filtering
# all keys can be used to filter eddies
PARAMS = {
	'lon': 0,
	'lat': 1,
	'tidx': 2,
	'eidx': 3,
	'age': 4,
	'lifetime': 5,
	'state': 6,
	'cyc': 7,
	'num_px': 8,
	'area': 9,
	'in_subdomain': 10
}


def apply_eddy_filter(filter, eddy_data, used_tracks):
	# check if parameter is valid
	assert filter[0] in PARAMS, 'Unknown property '+filter[0]
	# get value of parameter
	val = eddy_data[PARAMS[filter[0]]]
	# values can be None, convert them to np.nans
	val = np.array([i if i is not None else np.nan for i in val])
	# if age or lifetime, convert nanoseconds to days
	val = val if filter[0] not in ('age', 'lifetime') else val / (1e9 * 3600 * 24)
	# parse operator
	if filter[1] == '>':
		return val > filter[2]
	elif filter[1] == '>=':
		return val >= filter[2]
	elif filter[1] == '<':
		return val < filter[2]
	elif filter[1] == '<=':
		return val <= filter[2]
	elif filter[1] == '==':
		return val == filter[2]
	elif filter[1] == '!=':
		return val != filter[2]
	elif filter[1] == 'unique':
		return ~np.isin(val.astype(int), np.array(list(used_tracks)))
	else:
		raise RuntimeError('Unknown operator', filter[1])


def bbox_axis(bool_map, axis, num_radii=1, radius=None):
	"""
	Get a slice describing the bbox for a bool map along a given axis
	"""
	bm = np.sum(bool_map, axis=axis) > 0
	# get first and last position of True
	start = np.argmax(bm)
	end = len(bm) - np.argmax(bm[::-1])
	# calculate the radius as half the width
	r = radius if radius is not None else (end - start) / 2
	r_l = int(np.floor(r))  # in case of r = x.5 this needs to be x
	r_r = int(np.ceil(r)) # in case of r = x.5, this needs to be x + 1
	# center = start + r_l
	
	center = int(np.average(np.arange(bool_map.shape[(axis + 1) % 2]).astype(float), weights=np.sum(bool_map, axis=axis).astype(float)))

	return slice(center - num_radii*r_l, center + num_radii*r_r)


def bool_map_to_bbox(bool_map, num_radii=1):
	"""
	Get bbox for both axis
	"""
	# TODO: r can be calculated from the actual area (see below), but for some reason half the width is used (see bbox_axis)
	# a = np.count_nonzero(bool_map)
	# r = int(np.sqrt(a / np.pi))
	r = None
	return (bbox_axis(bool_map, axis=1, num_radii=num_radii, radius=r), bbox_axis(bool_map, axis=0, num_radii=num_radii, radius=r))


def get_eddy_np_data(eddies):
	"""
	Convert eddy data to numpy array (keep ordering given by PARAMS dict).
	"""
	data = [None for _ in PARAMS]
	for param in PARAMS:
		idx = PARAMS[param]
		data[idx] = eddies[param].values
	return np.array(data)


def interpolate(composite, num_px=51, method='nearest'):
	"""
	Interpolate rectangle to quadratic grid with given num_px
	"""
	# get distance from center in x direction for source rectangle
	x = np.tile(np.linspace(-1, 1, composite.shape[2]), (composite.shape[1], 1)).reshape(-1)
	# get distance from center in y direction for source rectangle
	y = np.tile(np.linspace(-1, 1, composite.shape[1]).reshape(-1, 1), (1, composite.shape[2])).reshape(-1)
	# get distance from center in x direction for target square
	tx = np.tile(np.linspace(-1, 1, num_px), (num_px, 1)).reshape(-1)
	# get distance from center in y direction for target square
	ty = np.tile(np.linspace(-1, 1, num_px).reshape(-1, 1), (1, num_px)).reshape(-1)
	# interpolate every depth
	return np.array([
		griddata((x,y), comp.reshape(-1), (tx, ty), method=method).reshape(num_px, num_px) for comp in composite
	])
		


def get_composite_data(path_out, eddies, ds, var_name, filters, climatology=None, num_radii=1, fully_inside=False, drop_nans=False, days_around=30):

	assert (filters[-1][2] == 1 or filters[-1][2] == 2) and filters[-1][0] == 'cyc'

	# arrays for results
	composites = []
	composite_slices = []
	composite_eidxs = []
	composite_tracks = defaultdict(list)
	# keep track of the used tracks 
	used_tracks = set()
	# shardening
	shard_idx = 0

	# load all eddy data (should not be so much actually)
	eddy_np_index = get_eddy_np_data(eddies)

	# define a filter function
	def filter_fn(data, used_tracks):
		# first every eddy passes
		mask = np.ones(data.shape[1]).astype(bool)
		for filter in filters:
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				mask = np.logical_and(
					mask, apply_eddy_filter(filter, data, used_tracks)  # apply filter and merge as logical and
				)
		# return eddy idxs which fulfill filters
		return data[PARAMS['eidx'], mask]

	total_eddies = 0

	# flush data to file and reset arrays
	def flush_data():
		nonlocal shard_idx, composites, composite_eidxs, composite_slices, composite_tracks, total_eddies
		out_path = path_out.replace('.npz', '-{:05d}.npz'.format(shard_idx))
		print('Flushing {:d} eddies to {:s}, {:d} in total'.format(len(composites), out_path, total_eddies))
		np.savez(
			out_path,
			composites=composites,
			composite_eidxs=composite_eidxs,
			composite_slices=composite_slices,
			composite_tracks=composite_tracks,
			filters=filters
		)
		composites = []
		composite_eidxs = []
		composite_slices = []
		composite_tracks = defaultdict(list)
		shard_idx += 1  # increase shard number for next call

	# load climatology
	clim_data = None
	if type(climatology) == xr.Dataset:
		print('Start loading climatology, this takes some time...')
		clim_data = climatology[var_name+'_b'].values
		print('Climatology loaded')
	else:
		clim_data = climatology  # numpy array and None

	days_without_eddies = 0

	print('WARNING: skipping first two years')

	# loop days
	for t_idx, t_obj in enumerate(eddies.time.values):

		# skip first two years
		if t_obj.year < 3 or t_obj.year > 5:
			continue

		# get meta information
		meta = eddy_np_index[:, t_idx, :]
		# get matching eddies
		matching_eddies = filter_fn(meta, used_tracks)

		# early return
		if len(matching_eddies) == 0:
			days_without_eddies += 1
			continue

		# get data
		data = np.squeeze(ds[var_name].sel(time=t_obj, method='nearest').values.copy())
		if len(data.shape) == 2:
			data = np.expand_dims(data, 0)
		if climatology is not None and len(clim_data.shape) == 3:
			clim_data = None if clim_data is None else np.expand_dims(clim_data, 1)
			
		if climatology is not None:
			# get doy and calculate the doys affected
			doys = get_doys(t_obj, ds, days_around)
			doys_weights = get_triangular_weights(doys)
			# substract the climatology from each grid point
			with warnings.catch_warnings():
				warnings.simplefilter("ignore")
				clim_mean = np.nansum(clim_data[doys] * doys_weights[:, None, None, None], axis=0)
			data -= clim_mean  # clim_data[doys].mean(axis=0)

		# eddy index map and lists
		edix_map = eddies.eidx_map.isel(time=t_idx).values
		eidx_list = eddies.eidx.isel(time=t_idx).values
		tidx_list = eddies.tidx.isel(time=t_idx).values

		timestep_tracks = {}

		# loop matching eddies
		for eddy_idx in matching_eddies:
			# assert np.count_nonzero(eddy_np_index[3] == eddy_idx) == 1

			# get lfd index
			lfd = np.argmax(eidx_list == eddy_idx)

			# create bool mask and get its bbox
			bool_mask = edix_map == eddy_idx
			if np.count_nonzero(bool_mask) == 0:
				continue
				
			slices = bool_map_to_bbox(bool_mask, num_radii=num_radii)

			# check if the data needs to be padded (because bbox outside)
			mw = max(
				-slices[0].start, # start negative means that the bbox begins before data
				-slices[1].start, 
				slices[0].stop - data.shape[1], # stop > shape means that the bbox exceeds data
				slices[1].stop - data.shape[2], 
				0  # default
			)
			
			if mw > 0 and fully_inside: # if mw > 0 the eddy is not fully inside
				continue

			# pad data, bool mask and adjust slices
			data_padded = np.pad(data, ((0, 0), (mw, mw), (mw, mw)), constant_values=np.nan)
			slices_padded = (slice(slices[0].start + mw, slices[0].stop + mw), slice(slices[1].start + mw, slices[1].stop + mw))
			
			# get data
			comp = data_padded[:, slices_padded[0], slices_padded[1]]
			# skip if it is too small
			if comp.shape[1] < 4 and comp.shape[2] < 4:
				continue
			# interpolate it
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				comp_interpolated = interpolate(comp)

			# if drop_nans and we have some nans in the interpolation
			if drop_nans and np.count_nonzero(np.isnan(comp_interpolated)) > 0:
				continue

			total_eddies += 1

			# save composite
			in_shard_idx = len(composites)
			composites.append(comp_interpolated)
			composite_slices.append(slices)
			composite_eidxs.append(eddy_idx)
			track_id = int(tidx_list[lfd])
			used_tracks.add(track_id)
			composite_tracks[track_id].append(in_shard_idx)

		print('Processing {}: {:d} composites'.format(t_obj, len(composites)), end='\r')
		
		# flush data if enough composites
		if len(composites) > 5000:
			flush_data()

	
	print(total_eddies)
	
	# flush last shard
	if len(composites) > 0:
		flush_data()

	return shard_idx


if __name__ == "__main__":

	import argparse
	import progressbar

	class FilterParser(argparse.Action):
		def __init__(self, option_strings, dest, nargs=None, **kwargs):
			self._nargs = nargs
			super(FilterParser, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

		def __call__(self, parser, namespace, values, option_string=None):
			filters = []

			def add_filter(operator, kv):
				var_name, value = kv.split(operator)
				assert var_name in PARAMS, 'Could not find variable ' + var_name
				value = int(value) if '.' not in value else float(value)
				filters.append((var_name, operator, value))

			for kv in values:
				found = False
				for operator in ('>=', '>', '<=', '<', '==', '!='):
					if operator in kv:
						found = True
						add_filter(operator, kv)
						break
				if kv == 'unique_track':
					found = True
					filters.append(('tidx', 'unique'))
				if not found:
					raise RuntimeError('Could not parse filter', kv)
			
			setattr(namespace, self.dest, filters)

	# Define arguments
	parser = argparse.ArgumentParser()
	get_dataset = dataset_from_args(parser)

	parser.add_argument("-o", "--output", type=str, help="Output path", required=True)
	parser.add_argument("--filter", action=FilterParser, nargs='+', required=True)
	parser.add_argument('-e', '--eddies', type=str, help="Path to eddies", required=True)
	parser.add_argument('-r', '--num_radii', type=int, help="Radius", default=1)
	parser.add_argument('--fully-inside', action='store_true', help="Require that the eddies are fully inside the data domain", default=False)
	parser.add_argument('--drop-nans', action='store_true', help="Drop eddy if it contains some nans", default=False)
	parser.add_argument('--days-around', type=int, help='Number of days around rolling mean', default=30)
	parser.add_argument('-c', '--climatology', type=str, help="Path to climatology")

	args = parser.parse_args()
	print('Parsed filters', args.filter)

	# replace cyc filters, because we do both, cyclones and anticyclones, anyway
	args.filter.append([
		'cyc', '==', 0
	])
	
	# get initial dataset
	ds = get_dataset(args)
	variables = list([var for var in ds.variables])
	ds.close()

	for variable in variables:
		# reload data
		eddies = xr.open_dataset(args.eddies)
		ds = get_dataset(args)
		
		# check if data has enough dimensions
		if len(ds[variable].shape) < 3:
			continue
		
		print('Processing variable', variable)

		# load climatology (i.e. the values!)
		climatology = None 
		if args.climatology is not None:
			climatology = open_dataset(args.climatology, eta_rho_slice=args.eta_rho, xi_rho_slice=args.xi_rho, s_rho_slice=args.s_rho)
			climatology = climatology[variable+'_b'].values

		## cyclones
		assert args.filter[-1][0] == 'cyc'
		args.filter[-1][2] = 2
		# set up out path
		out_path = args.output.replace('.npz', '-cycl-{:s}.npz'.format(variable))
		# do the calculation
		num_shards = get_composite_data(out_path, eddies, ds, variable, args.filter, num_radii=args.num_radii, fully_inside=args.fully_inside, drop_nans=args.drop_nans, climatology=climatology, days_around=args.days_around)
		print('Done. Data saved to {:s} in {:d} shards'.format(out_path, num_shards))

		## anticyclones
		assert args.filter[-1][0] == 'cyc'
		args.filter[-1][2] = 1
		# set up out path
		out_path = args.output.replace('.npz', '-anti-{:s}.npz'.format(variable))
		# do the calculation
		num_shards = get_composite_data(out_path, eddies, ds, variable, args.filter, num_radii=args.num_radii, fully_inside=args.fully_inside, drop_nans=args.drop_nans, climatology=climatology, days_around=args.days_around)
		print('Done. Data saved to {:s} in {:d} shards'.format(out_path, num_shards))

		# close to drop in memory data
		eddies.close()
		ds.close()