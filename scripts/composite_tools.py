#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020
# Project: master-thesis


import sys                                                                                                                                                    
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
import glob
from romstools.cmap import W2G, W2G_r, DIFF, DIFF_r, get_step_cmap
from scipy import ndimage
import cartopy.crs as ccrs
from collections import defaultdict

from romstools.plot import plot_block
from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.colors as mplc

import warnings

# defines the size of the composites
# TODO: get this from data
COMPOSITE_DIM = 51
# the core is given as -R to R (R = eddy radii). The 51 pixels are ranging from -2R to 2R
COMPOSITE_CORE = slice(12, 37)


def im(data):
	"""
	imshow puts eta (first axis) on y-axis inverts the y-axis resulting in a plot like
	W   |
		|
	eta |
		|
	E   v -------->
		  N  xi  S
		  
	So flipping brings us to
	W   |
		|
	eta |
		|
	E   v -------->
		  S  xi  N
	
	And rotation to
	N   |
		|
	xi  |
		|
	S   v -------->
		  E  xi  W
	"""
	return np.rot90(np.flip(data, axis=1), k=1)


##############################################
#  Filtering means to filter out eddies
#  to use from a shard. All these functions
#  return a set of indices which correspond
#  to those eddies in a shard which should be 
#  considered for calculations.
##############################################

def filter_by_chl(path, lt0, thickness, eddies_idx):
	"""
	Get only those instances from a shard which have a value larger 0 (or smaller 0).
	This can be useful to filter eddies based on the sign of their anomaly. However, this was actually never used :)
	"""
	# load data
	d = np.load(path, allow_pickle=True)
	# calc chl
	data = d['composites']
	data *= thickness[np.newaxis, :, np.newaxis, np.newaxis]
	data = data.sum(axis=1)
	# get eidxs
	eidxs = d['composite_eidxs']
	assert len(eidxs) == len(data)
	# keep track of corresponding idxs
	in_shard_idxs = []
	# loop all data
	for in_shard_idx, eddy_idx in enumerate(eidxs):
		# get mask for eddy data
		mask = eddies_idx == eddy_idx
		assert np.count_nonzero(mask) == 1
		# condition
		if (lt0 and np.nanmean(data[in_shard_idx, COMPOSITE_CORE, COMPOSITE_CORE]) > 0) or \
		   (not lt0 and np.nanmean(data[in_shard_idx, COMPOSITE_CORE, COMPOSITE_CORE]) < 0):
			in_shard_idxs.append(in_shard_idx)
	
	return set(in_shard_idxs)


def filter_by_mask(path, mask, eddies_idxmap):
	"""
	Get only those instances from a shard which intersect with a given boolean map.
	"""
	# load data
	d = np.load(path, allow_pickle=True)
	# get eidxs
	eidxs = d['composite_eidxs']
	# get intersections
	within_eddies = set(np.unique(
		eddies_idxmap[:, mask]
	))

	in_shard_idxs = []
	# loop all data
	for in_shard_idx, eddy_idx in enumerate(eidxs):
		if eddy_idx in within_eddies:
			in_shard_idxs.append(in_shard_idx)
	return set(in_shard_idxs)


def filter_by_track(path, track_id):
	"""
	Get only those instances that belong to a given track.
	"""
	# load data
	d = np.load(path, allow_pickle=True)
	# get eidxs
	tracks = d['composite_tracks'].item()
	if track_id in tracks:
		return set(tracks[track_id])
	else:
		return set([])


def filter_by_dist(path, dist_map, ge, lt, eddies_idxmap):
	"""
	Get only those instances that are within a given range of distance from coast.
	"""
	mask = np.logical_and(dist_map >= ge, dist_map < lt)
	return filter_by_mask(path, mask, eddies_idxmap)


def combine_filters(fl1, fl2):
	"""
	Combine two filter function results
	"""
	assert len(fl1) == len(fl2)
	r = []
	for i in range(len(fl1)):
		r.append(
			fl1[i].intersection(fl2[i])
		)
	return r


##########################################
#  Splitting means adding a new
#  dimension to the results corresponding
#  to the splitting attribute
##########################################

def split_by_eddiesX(path, eddies_idx, eddies_val):
	# X can be age or doy
	# load data
	d = np.load(path, allow_pickle=True)
	# get eidxs
	eidxs = d['composite_eidxs']
	# keep track of corresponding idxs
	in_shard_idxs = defaultdict(set)

	# loop all data
	for in_shard_idx, eddy_idx in enumerate(eidxs):
		# get mask for eddy data
		mask = eddies_idx == eddy_idx
		assert np.sum(mask) == 1
		if len(mask.shape) > len(eddies_val.shape):
			mask = np.sum(mask, axis=1) > 0
		e_val = int(eddies_val[mask])
		in_shard_idxs[e_val].add(in_shard_idx)
	
	return dict(in_shard_idxs)


####################################################
#  Functions to load and aggregate data of a shard
####################################################


def get_shard_mean(path, list_in_shard_idxs, absolute_value=False):
	"""
	Calculate the mean eddy of a shard (averaged over eddy instances)
	"""
	d = np.load(path, allow_pickle=True)
	composites = d['composites']
	r = []
	for in_shard_idxs in list_in_shard_idxs:
		# check if instance should be used
		composites_f = composites if in_shard_idxs is None else composites[np.array(list(in_shard_idxs)).astype(int)]
		# absolute value?
		if absolute_value:
			composites_f = np.abs(composites_f)
		# add nothing if f has no shape for some reason
		if composites_f.shape[0] == 0:
			r.append((np.zeros(composites_f.shape[1:]), 0))
		# else add mean and save number of used eddies (to combine the mean of several shards)
		else:
			mean_composites = np.nanmean(composites_f, axis=0)
			r.append((mean_composites, composites_f.shape[0]))
	return r


def get_shard_flux(path_val1, path_val2, list_in_shard_idxs):
	"""
	Calculate a flux of a shard (averaged over eddy instances). It is very similar to get_shard_mean, but it opens two shards
	and always multiplies variable1 with variable2
	"""
	d = np.load(path_val1, allow_pickle=True)
	composites1 = d['composites']
	d = np.load(path_val2, allow_pickle=True)
	composites2 = d['composites']
	# check if the same
	assert composites1.shape == composites2.shape
	
	r = []
	for in_shard_idxs in list_in_shard_idxs:
		# get slice based on filter
		s = slice(None, None) if in_shard_idxs is None else np.array(list(in_shard_idxs)).astype(int)
		composites1_f = composites1[s]
		composites2_f = composites2[s]
		len_comp1_f = composites1_f.shape[0]
		# if no items match the filter
		if len_comp1_f == 0:
			r.append((np.zeros(composites1_f.shape[1:]), 0))
		else:
			flux = composites1_f * composites2_f
			mean_composites = np.nanmean(flux, axis=0)
			r.append((mean_composites, len_comp1_f))
	return r


def get_mean(base_path, var, list_list_in_shard_idxs, absolute_value=False):
	"""
	Calculate the results for all shards (flux or mean). TODO: name is misleading
	"""
	res = None
	total_items = None
	# get all shards and fetch number of shards
	var_glob = var if type(var) == str else var[0]
	num_shards = len(
		glob.glob(base_path + var_glob + '*.npz')
	)

	def get_shard_res(shard_idx):
		# get list of eddies to use for this shard
		list_in_shard_idxs = [None if lisi is None else lisi[shard_idx] for lisi in list_list_in_shard_idxs]
		if type(var) == str:  # single variable, just get mean
			path = base_path + var + '-{:05d}.npz'.format(shard_idx)
			return get_shard_mean(path, list_in_shard_idxs, absolute_value=absolute_value)
		else:
			assert len(var) == 2  # two variables, get flux
			path1 = base_path + var[0] + '-{:05d}.npz'.format(shard_idx)
			path2 = base_path + var[1] + '-{:05d}.npz'.format(shard_idx)
			return get_shard_flux(path1, path2, list_in_shard_idxs)

	# loop shards
	for shard_idx in range(num_shards):
		shard_res = get_shard_res(shard_idx)
		# do some checks
		assert len(shard_res) == len(list_list_in_shard_idxs)
		assert len(shard_res[0]) == 2
		# init res and total_items
		if res is None:
			res = [r[0] * r[1] for r in shard_res]
			total_items = [r[1] for r in shard_res]
		# save for each res the weighted mean
		else:
			for i, r in enumerate(shard_res):
				res[i] += (r[0] * r[1])
				total_items[i] += r[1]
	# calc the mean by dividing by the number of instances used for the mean (its weighted)
	for i in range(len(list_list_in_shard_idxs)):
		res[i] /= total_items[i]
	
	return res, total_items


################
#  Plotting
################

def plot_chl(ax, val, zeta=None, v=None, num_levels=21, cmap=DIFF_r, title='', ssh_range=np.linspace(-0.05, 0.05, 11)):
	"""
	The function plots 2D data (originating from composites, e.g. surface chl) and configures the figure accordingly.
	"""
	# levels for ssh
	levels = ssh_range
	# init vmin and vmax
	vmin, vmax = None, None
	if type(v) == float or type(v) == int:
		vmin, vmax = -v, v
	if type(v) == tuple:
		vmin, vmax = v
	# plot data
	cax = ax.imshow(im(val), cmap=get_step_cmap(cmap, num_levels), vmin=vmin, vmax=vmax)
	# plot contours (SSH)
	cax_zeta = None
	if zeta is not None:
		cax_zeta = ax.contour(im(zeta), vmin=np.min(ssh_range), vmax=np.max(ssh_range), cmap='PiYG', levels=levels)
	# title
	ax.set_title(title)
	# add grid (i.e. show eddy radii)
	ax.axhline(COMPOSITE_DIM/2, 0, 1, color='k', lw=1)
	ax.axvline(COMPOSITE_DIM/2, 0, 1, color='k', lw=1)
	ax.axhline(COMPOSITE_DIM/4, 0, 1, color='k', ls='--', lw=1)
	ax.axvline(COMPOSITE_DIM/4, 0, 1, color='k', ls='--', lw=1)
	ax.axhline(3*COMPOSITE_DIM/4, 0, 1, color='k', ls='--', lw=1)
	ax.axvline(3*COMPOSITE_DIM/4, 0, 1, color='k', ls='--', lw=1)
	# adjust ticks
	ax.set_xticks([0, COMPOSITE_DIM/4, COMPOSITE_DIM/2, 3*COMPOSITE_DIM/4, COMPOSITE_DIM])
	ax.set_xticklabels(['-2R', '-R', '0', 'R', '2R'])
	ax.set_yticks([0, COMPOSITE_DIM/4, COMPOSITE_DIM/2, 3*COMPOSITE_DIM/4, COMPOSITE_DIM])
	ax.set_yticklabels(['-2R', '-R', '0', 'R', '2R'])
	# labels
	ax.set_xlabel('West          $\\eta$          East')
	ax.set_ylabel('South          $\\xi$          North')

	return cax, cax_zeta


def plot_3d(ax, zlevels, val, zeta, v=None, num_levels=21, surface_range=slice(0, 2), core_range=slice(23, 28), zoff=3, cmap=DIFF_r):
	"""
	Wrapper function for plot_block. But actually it is just used to define the default values accordingly :)
	"""
	# init vmin and vmax
	vmin, vmax = None, None
	if type(v) == float or type(v) == int:
		vmin, vmax = -v, v
	if type(v) == tuple:
		vmin, vmax = v
	# call plot_block
	cax1, cax2, _ = plot_block(ax, val, 
			   vmin=vmin, vmax=vmax, 
			   eta_range=core_range, xi_range=core_range, surface_range=surface_range, 
			   surface_contours=zeta, 
			   surface_contours_kwargs={'cmap': 'PiYG', 'vmin': -0.05, 'vmax': 0.05, 'levels': np.linspace(-0.05, 0.05, 11)},
			   z_level=zlevels, zoff=zoff, off=0.05, r=2, levels=num_levels,
			   cmap=get_step_cmap(cmap, 21)
	)

	return cax1, cax2


def mono_dipol_structure(data, num_levels=None):
	"""
	Decompose 2D data into a monopole and a residual. TODO: name is misleading.
	"""
	# calculate the radial mean
	sx, sy = data.shape
	num_levels = num_levels if num_levels is not None else max(sx, sy)
	# ... create a circular boolean map
	X, Y = np.ogrid[0:sx, 0:sy]
	r = np.hypot(X - sx/2, Y - sy/2)
	# ... bin data
	rbin = (num_levels* r/r.max()).astype(np.int)
	# ... get mean
	radial_mean = ndimage.mean(data, labels=rbin, index=np.arange(1, rbin.max() +1))
	
	# now we have to convert radial_mean (1D) back to 2D
	radial_mean_pop = np.zeros_like(data)
	# TODO: there has to be a smarter way!
	for idx in range(num_levels):
		radial_mean_pop[rbin == idx + 1] = radial_mean[idx]
	
	# return radial mean and residual
	return radial_mean_pop, data - radial_mean_pop


#######################
#  Composite Manager
#######################

def int_or_none(v):
	"""
	Utility function for parsing slices
	"""
	if v == '' or v == 'none':
		return None
	else:
		return int(v)


class CompositeManager:
	
	def __init__(self, path_levels, path_grid_data, path_eddies, path_composite_base, num_depths_chl=20, three_to_five=True):
		
		# load levels and thickness with correct slicing
		# TODO: we hardcoded here, that we only used every second depth level (except for CHL, where we used all but only took the upper 20 levels)
		levels = xr.open_dataset(path_levels)
		self.thickness_chl = levels.thickness_z[:num_depths_chl].values
		self.thickness = levels.thickness_z[::2].values
		self.zlevels = levels.z_level.values[::2]
		self.zlevels_chl = levels.z_level.values[:num_depths_chl]
		
		# load distance map and analysis domain
		data = np.load(path_grid_data)
		self.distance_map = data['distance_map']
		self.gruber_mask = data['gruber_mask']
		
		# load eddy data and assign as object attributes
		eddy_data = xr.open_dataset(path_eddies)
		self.eddies_idx = eddy_data.eidx.values
		self.eddies_lon = eddy_data.lon.values
		self.eddies_lat = eddy_data.lat.values
		self.eddies_age = (eddy_data.age.values * 1e-9 / (3600 * 24)).astype(int)  # as days instead of nanoseconds
		self.eddies_track = eddy_data.tidx.values
		self.eddies_idxmap = eddy_data.eidx_map.values
		
		# for some reason, pactcs15 changed the doys. So we rewrite all doys to be even
		# TODO: this is hardcoded for bidaily!
		self.eddies_doy = eddy_data.doy.values
		print('Warning: fix doys')
		self.eddies_doy = np.array([doy - (doy % 2) for doy in self.eddies_doy])
		
		# only use years three to five
		# TODO: this is hardcoded for the present integration time
		if three_to_five:
			self.eddies_doy[:364] = -1
			self.eddies_doy[910:] = -1
			print('WARNING: only use yr 3 to 5 ({:d}/{:d})'.format(np.count_nonzero(self.eddies_doy != -1), self.eddies_doy.shape[0]))
		
		# set up paths for composites
		self.path_composite_base = path_composite_base
		self.base_path = {
			'cycl': path_composite_base+'cycl-',
			'anti': path_composite_base+'anti-'
		}
		
		# calculate the number of shards
		self.num_shards = {
			'cycl': len(glob.glob(self.get_basepath('zeta', 'cycl', '*.npz'))),
			'anti': len(glob.glob(self.get_basepath('zeta', 'anti', '*.npz')))
		}
		
		# data storage for calculated data
		self.data = {}
		# keep track of the number of used eddies
		self.num_items = {}
		
		self.filters = {}
		self.splitter = {}
		
		
	def get_basepath(self, var, cycl, shard_placeholder='{:05d}.npz'):
		# construct the path for a given variable and polarity
		return self.base_path[cycl]+var+'-'+shard_placeholder
		
		
	def get_filter(self, cycl, filter):
		"""
		Get filter data, i.e. list of list of indices (first list for shards, second list for in-shard indices of eddies to use).
		"""
		
		if filter == '' or filter == 'none':
			return None
		
		# filter by CHL anomaly (positive or negative)
		elif filter[:3] == 'chl':
			pos_anomaly = filter[3] == '+'  # positive or negative
			filter_name = 'chlanomaly_'+cycl+'_'+('pos' if pos_anomaly else 'neg')  # unique identifier for the filter (for caching)
			if filter_name not in self.filters:
				# construct filter
				self.filters[filter_name] = [
					filter_by_chl(self.get_basepath('TOT_CHL', cycl).format(i), pos_anomaly, self.thickness_chl, self.eddies_idx) for i in range(self.num_shards[cycl])
				]
			return self.filters[filter_name]
		
		# filter by distance to coast
		elif filter[:4] == 'dist':
			# get meta data for filter, i.e. range
			min_dist, max_dist = filter[4:].split(':')
			min_dist = int(min_dist)
			max_dist = int(max_dist)
			filter_name = 'dist_'+cycl+'_{:05d}-{:05d}'.format(min_dist, max_dist)  # unique identifier for the filter (for caching)
			if filter_name not in self.filters:
				# construct filter
				self.filters[filter_name] = [
					filter_by_dist(self.get_basepath('zeta', cycl).format(i), self.distance_map, min_dist, max_dist, self.eddies_idxmap) for i in range(self.num_shards[cycl])
				]
			return self.filters[filter_name]
		
		# only use eddies in analysis domain
		elif filter[:6] == 'gruber':
			filter_name = 'grubermask_'+cycl  # unique identifier for the filter (for caching)
			if filter_name not in self.filters:
				# construct filter
				self.filters[filter_name] = [
					filter_by_mask(self.get_basepath('zeta', cycl).format(i), self.gruber_mask, self.eddies_idxmap) for i in range(self.num_shards[cycl])
				]
			return self.filters[filter_name]

		# only use eddies of a specific track
		elif filter[:5] == 'track':
			# parse track id from meta data
			track_id = int(filter[5:])
			print('Search track', track_id)
			filter_name = 'track_{:d}'.format(track_id)  # unique identifier for the filter (for caching)
			if filter_name not in self.filters:
				# construct filter
				self.filters[filter_name] = [
					filter_by_track(self.get_basepath('zeta', cycl).format(i), track_id) for i in range(self.num_shards[cycl])
				]
			return self.filters[filter_name]

		else:
			raise RuntimeError('Unknown filter', filter)
		
		
	def __getitem__(self, key):
		# This function dynamically creates eddy composites, based on the provided key. The structure of key has to be the following:
		# VARIABLE POLARITY (FILTER) (SPLIT) (REDUCTION) (RMSD)
		# - VARIABLE: specify the variable to use
		# - POLARITY: specify the polarity (cycl or anti) to use
		# - FILTER: Filter the eddies to use for the composite. Possible filters are distX:Y (X is minimum distance, Y maximum), gruber (analysis domain) and trackZ (Z is the track index).
		#           Filters can be combined using &. If you do not want to provide a filter type none
		# - SPLIT: the composite can be splitted (meaning that there is an individual composite for every possible value of SPLIT). doy and age are supported.
		#          The range of values to consider can be provided as a slice.
		#          For example: age5:15 would return 10 composites. The first composite is only based on eddies with an age of 5 days, the second only on eddies with age of 6 days, ...
		#          This is extremely useful when you want to filter composites based on season (-> doy). If you do not want to provide a split type none
		# - REDUCTION: define how depth levels should be reduced. Possible values are none, avg, sum, integrate (weighted sum based on layer thickness) or an integer specifying the depth level to use
		# - RMSD: when providing rmsd, the absolute value of the variable is used. TODO: misleading name
		# Example: 'temp cycl gruber&dist200:800 doy30:60' would return the mean temperature anomaly of eddies during February in the analysis domain with a distance of 200 to 800km off the coast.
		#          The result dimension would be 30 (doys) x 47 (depth) x 51 x 51
		
		# first, check if result was cached
		if key not in self.data:
			commands = key.split(' ')
			# get required arguments
			var, cycl = commands[:2]
			# if flux, parse
			if var[0] == '(':
				var = tuple(k.strip() for k in var[1:-1].split(','))
			# set up filtering
			filter = None
			
			# do we have a filter?
			if len(commands) >= 3 and commands[2] != 'none':
				# set up filters
				filters = [self.get_filter(cycl, f) for f in commands[2].split('&')]
				# reduce filters by logical and
				filter = filters[0]
				for i in range(1, len(filters)):
					filter = combine_filters(filter, filters[i])
			
			# do we need to split
			if len(commands) >= 4 and commands[3] != 'none':
				split_by = commands[3]
				assert split_by[:3] in ('age', 'doy')
				# get the correct data to use for split
				split_x = self.eddies_age if split_by[:3] == 'age' else self.eddies_doy
				# set up a splitter
				splitter_name = cycl+'_'+split_by
				if splitter_name not in self.splitter:
					# split all data
					self.splitter[splitter_name] = [
						split_by_eddiesX(self.get_basepath('zeta', cycl).format(i), self.eddies_idx, split_x) for i in range(self.num_shards[cycl])
					]
					
				all_filters = []
				# create array of items to consider
				split_slice_def = map(int_or_none, split_by[3:].split(':'))
				keys = set([]).union(*[set(lol.keys()) for lol in self.splitter[splitter_name]])
				# sort the keys and split
				keys = np.unique(list(keys))[slice(*split_slice_def)]

				print('Split keys:', keys)

				for yo in keys:
					# create a new filter by combining with FILTER from above
					new_filter = combine_filters(filter, [
						set([]) if yo not in splitter_shard else splitter_shard[yo] for splitter_shard in self.splitter[splitter_name]
					])
					# and add to filters
					all_filters.append(new_filter)
				
				# now we have a filter for every possible value of SPLIT.
				# all the filters already have the filter based on FILTER already merged
				filter = all_filters	
			else:
				# to get the dimensions right
				filter = [filter]
					
			# should we get absolute value
			as_absolute_value = len(commands) >= 6 and commands[5] == 'rmsd'
			if as_absolute_value:
				print('Use absolute value')

			# get 3D data
			data, num_items = get_mean(
				self.base_path[cycl], var, filter, absolute_value=as_absolute_value
			)
			data = np.array(data)
			
			# should we reduce depth dimension?
			if len(commands) >= 5 and commands[4] != 'none':
				reducing_method = commands[4]

				if reducing_method == 'integrate':
					# integration
					thick = self.thickness if len(self.thickness) == data.shape[1] else self.thickness_chl
					data = (data * thick[np.newaxis, :, np.newaxis, np.newaxis]).sum(axis=1)
				elif reducing_method[:3] == 'avg' or reducing_method[:3] == 'sum':
					# sum or average
					fn = np.nanmean if reducing_method[:3] == 'avg' else np.nansum
					mean_slice_def = map(int_or_none, reducing_method[3:].split(':'))
					s = slice(*mean_slice_def)
					data = fn(data[:, s], axis=1)
				else:
					# take a specific depth index
					data = data[:, int(reducing_method)]

			# keep data in cache
			self.data[key] = data
			# save the number of used eddies
			self.num_items[key] = num_items
			
			print('Done calculating', key)

		# return
		return self.data[key]