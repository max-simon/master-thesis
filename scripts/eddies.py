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
from romstools.utils import get_area_map
import netCDF4 as nc
from cftime import date2num
from functools import partial
from geopy.distance import geodesic

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

def get_date_track_index(path_anti, path_cycl, ds, freq):
	"""
	Create an index of tracks per day
	"""
	tracks_anticycl = sio.loadmat(path_anti, struct_as_record=False, squeeze_me=True)['anticyclonic_tracks']
	tracks_cycl = sio.loadmat(path_cycl, struct_as_record=False, squeeze_me=True)['cyclonic_tracks']

	# build date-track-index, for each time_index there is a set of track indices
	date_track_index = defaultdict(set)
	# get times
	times = ds.time.values

	# loop each track
	track_idx_offset = 0
	for direction, data in [(1, tracks_anticycl), (2, tracks_cycl)]:
		for track_idx, track in enumerate(data):
			# get number of frames
			num_frames, _ = track.shape
			# calculate time index of start and end
			time_idx_start = int((data[track_idx][0, 2] - 1)/freq)
			time_idx_end = int((data[track_idx][-1, 2] - 1)/freq)
			# get lifetime
			lifetime = np.around((times[time_idx_end] - times[time_idx_start]).total_seconds() / (3600 * 24))
			if abs(lifetime - int((num_frames-1)*freq)) > 2:
				print('WARNING: determined lifetime ({:.0f}) does not fit to number of frames ({:.0f})'.format(lifetime, num_frames))

			# add cycl-anticycl info and length of track info to data
			direction_info = np.ones((num_frames, 1)) * direction
			num_frames_info = np.ones((num_frames, 1)) * lifetime
			data[track_idx] = np.hstack([track, direction_info, num_frames_info])
			# add track to date-track-index
			for t in data[track_idx][:, 2]:
				date_track_index[t].add(track_idx_offset + track_idx)
			
			# check if eddies are distinct
			assert len(np.unique(track[:,2])) == len(track)
			
		# update track_idx_offset
		track_idx_offset += len(data)

	return date_track_index, np.hstack([tracks_anticycl, tracks_cycl])


def eddy_edge(bool_mask):
	"""
	Extract the edge of a boolean map
	"""
	edge_mask = np.zeros_like(bool_mask)
	# the idea is to loop over one axis and to get the start and end of 
	# the line at the second axis
	bm0 = np.sum(bool_mask, axis=1) > 0
	# in order to not loop overything, get start and end
	idx0_start = np.argmax(bm0)
	idx0_end = bool_mask.shape[0] - np.argmax(bm0[::-1])
	for i in range(idx0_start, idx0_end):
		idx_start = np.argmax(bool_mask[i])
		idx_end = bool_mask.shape[1] - np.argmax(bool_mask[i, ::-1])
		edge_mask[i, idx_start] = True
		edge_mask[i, idx_end-1] = True

	return edge_mask


def init_shard(out_path, ds, previous_shard, shard_nr):
	"""
	Create a new shard for eddy output
	"""
	print('\n')
	if previous_shard is not None:
		previous_shard.close()
		print('Closed previous shard.')
	if shard_nr is None:
		shard_nr = -1
	shard_nr += 1

	eddy_ds_path = out_path.replace('.nc', '-{:05d}.nc').format(shard_nr)

	# create file
	eddy_ds = nc.Dataset(eddy_ds_path, 'w')  # pylint: disable=no-member

	# copy dimensions
	eddy_ds.createDimension("time", None)
	eddy_ds.createDimension("lfd", None)
	eddy_ds.createDimension("eta_rho", ds.dims['eta_rho'])
	eddy_ds.createDimension("xi_rho", ds.dims['xi_rho'])

	# create enum type (must be done for each new shard)
	eddy_cycl_type = eddy_ds.createEnumType(int, 'eddy_cycl', EDDY_CYCL_DICT)
	eddy_state_type = eddy_ds.createEnumType(int, 'eddy_state', EDDY_STATE_DICT)
	
	# create dimensions
	## maps
	eddy_ds.createVariable("eidx_map", int, zlib=True, dimensions=('time', 'eta_rho', 'xi_rho'), fill_value=-1)
	eddy_ds.createVariable("tidx_map", int, zlib=True, dimensions=('time', 'eta_rho', 'xi_rho'), fill_value=-3)
	## coordinates
	eddy_ds.createVariable("lon", float, zlib=True, dimensions=('time', 'lfd'), fill_value=np.nan)
	eddy_ds.createVariable("lat", float, zlib=True, dimensions=('time', 'lfd'), fill_value=np.nan)
	eddy_ds.createVariable("distance_to_coast", float, zlib=True, dimensions=('time', 'lfd'), fill_value=np.nan)
	eddy_ds.createVariable("in_subdomain", int, zlib=True, dimensions=('time', 'lfd'), fill_value=-1)
	## time
	eddy_ds.createVariable("time", float, zlib=True, dimensions=('time',), fill_value=np.nan)
	eddy_ds.variables['time'].units = ds.time.attrs['units']
	eddy_ds.variables['time'].calendar = ds.time.attrs['calendar']
	eddy_ds.createVariable("doy", int, zlib=True, dimensions=('time',), fill_value=-1)
	## ids
	eddy_ds.createVariable("tidx", int, zlib=True, dimensions=('time', 'lfd'), fill_value=-3)
	eddy_ds.createVariable("eidx", int, zlib=True, dimensions=('time', 'lfd'), fill_value=-1)
	## eddy properties
	eddy_ds.createVariable("age", int, zlib=True, dimensions=('time', 'lfd'), fill_value=-1)
	eddy_ds.variables['age'].units = 'days'
	eddy_ds.createVariable("lifetime", int, zlib=True, dimensions=('time', 'lfd'), fill_value=-1)
	eddy_ds.variables['lifetime'].units = 'days'
	eddy_ds.createVariable("state", eddy_state_type, zlib=True, dimensions=('time', 'lfd'), fill_value=0)
	eddy_ds.createVariable("cyc", eddy_cycl_type, zlib=True, dimensions=('time', 'lfd'), fill_value=0)
	eddy_ds.createVariable("num_px", int, zlib=True, dimensions=('time', 'lfd'), fill_value=0)
	eddy_ds.createVariable("area", float, zlib=True, dimensions=('time', 'lfd'), fill_value=np.nan)
	eddy_ds.createVariable("amplitude", float, zlib=True, dimensions=('time', 'lfd'), fill_value=np.nan)
	eddy_ds.createVariable("distance", float, zlib=True, dimensions=('time', 'lfd'), fill_value=np.nan)
	
	print('Opened new shard at', eddy_ds_path)

	return eddy_ds, shard_nr, eddy_cycl_type, eddy_state_type


def run_ssh_debug(ds, time_index, ssh_path, freq):
	"""
	Check SSH value of dataset with SSH value extracted for eddy detection
	"""
	zeta = ds.zeta.isel(time=time_index).values.T * 100
	t = (freq * time_index) + 1 # matlab starts at 1

	try:
		# make sure, we have the same domain!
		ssh = sio.loadmat(ssh_path.format(t), struct_as_record=False, squeeze_me=True)['data']
		assert np.allclose(np.nan_to_num(ssh), np.nan_to_num(zeta)), (zeta.shape, ssh.shape)
	except ValueError:
		print('WARNING : could not check domain because file is not available', date)


def run_time_index(
	ds, 
	time_index, 
	freq, 
	area_map,
	distance_map, 
	date_track_idx, 
	tracks,
	eddy_paths,
	history,
	fillaments_ds=None,
	subdomain=None
):
	"""
	Run the data transformation procedure for one time index
	"""
	# fetch data
	data = ds.isel(time=time_index)
	date = data.time.values.item()
	doy = date.dayofyr - 1

	t = (freq * time_index) + 1 # matlab starts at 1

	# load fillaments if given
	fillaments = None if fillaments_ds is None else fillaments_ds.isel(time=time_index).fil_mask.values

	# get ssh for intensity
	zeta = data.zeta.values.T * 100

	# load cyclonic and anticyclonic eddies from
	# Faghmous detection
	eddies_cycl = sio.loadmat(eddy_paths['cycl'].format(t), struct_as_record=False, squeeze_me=True)['eddies']
	eddies_anti = sio.loadmat(eddy_paths['anti'].format(t), struct_as_record=False, squeeze_me=True)['eddies']

	# distinguish betwenn eddies which occur in a track (real or fake)
	# and eddies which were not assigned to any track
	eddies_from_track = set()
	eddies_not_from_track = set()
	# counter for fake eddies
	fake_eddies = 0

	# map where each pixel corresponds to the track_index of the eddy covering the track
	# positive values: tracks
	# -1: fillaments
	# -2: eddy without track
	# zero: else
	tidx = np.zeros((data.xi_rho.shape[0], data.eta_rho.shape[0])).astype(int)

	assert tidx.shape == area_map.shape, (area_map.shape, tidx.shape)
	assert distance_map.shape == tidx.shape, (distance_map.shape, tidx.shape)
	assert (subdomain is None) or (tidx.shape == subdomain.shape), (subdomain.shape, tidx.shape) 

	# map where each pixel corresponds to the overall_eddy_counter
	eidx = np.zeros_like(tidx).astype(int)

	# list of extracted eddies
	eddies = []
	eddy_dates = []

	# wrong classified
	wrong_classified = 0

	# loop over tracks found for this time
	for track_idx in date_track_idx[t]:
		# in order to not interfer with the zeros
		track_id = track_idx + 1
		
		# increase the counter
		history['overall_eddy_counter'] += 1
		# get entry of track corresponding to current day
		track = tracks[track_idx]

		# get the index of the current day inside the selected track
		track_day_idx = np.where(track[:, 2] == t)[0][0]
		# get the total number of days the track lasts
		assert (track[:, 7] == track[0, 7]).all()
		lifetime = track[0, 7]
		
		# check if we already used the track entry
		if track_day_idx in history['used_track_day_idxs'][track_id]:
			raise RuntimeError(track_day_idx)
		else:
			history['used_track_day_idxs'][track_id].add(track_day_idx)
		# get track information for this day
		track_day = track[track_day_idx]
		# get the eddy_index (starts at 1)
		eddy_idx_ = track_day[3]

		# presets
		track_day_type = EDDY_STATE_DICT['unknown']  # unknown
		
		eddy = None  # will be filled with a Eddy object
		eddy_idx = None  # eddy idx
		eddy_num_pixels = 0
		eddy_area = np.nan
		eddy_amplitude = np.nan
		eddy_in_subdomain = 0  # 0 is outside, 1 is inside, -1 is fake eddy
		eddy_distance_coast = np.nan

		if np.isnan(eddy_idx_):
			# this is a fake eddy
			track_day_type = EDDY_STATE_DICT['fake']  # fake eddy
			# get data of last eddy...
			eddy = history['last_eddy_index'][track_id]
			
			# ... and adjust/reset position
			eddy.Lat = track_day[0]
			eddy.Lon = track_day[1]
			eddy_type = EDDY_CYCL_DICT['anticyclonic'] if eddy.Cyc == 1 else EDDY_CYCL_DICT['cyclonic']

			# set full_eddy_id and increase counter
			fake_eddies += 1
			eddy_idx = -1 * fake_eddies
			eddy_in_subdomain = -1

		else: # this is a real eddy
			# decrease eddy_idx_ because should start at 0 in Python
			eddy_idx = eddy_idx_ - 1
			# load eddy data by index
			eddy = eddies_anti[eddy_idx] if track_day[6] == 1 else eddies_cycl[eddy_idx]
			# make sure, that we have indeed the correct eddy
			assert (eddy.Stats.PixelIdxList == track_day[4]).all(), (eddy.Stats.PixelIdxList, track_day[4])
			assert eddy.Lat == track_day[0] and eddy.Lon == track_day[1]
			eddy_type = EDDY_CYCL_DICT['anticyclonic'] if eddy.Cyc == 1 else EDDY_CYCL_DICT['cyclonic']  # 1 is antycyclonic in both

			# set the track_type (start == 1, end == 2, continue == 3)
			if track_day_idx == 0:
				track_day_type = EDDY_STATE_DICT['start']  # start
				history['eddy_start_times'][track_id] = date
			elif track_day_idx == len(track) - 1:
				track_day_type = EDDY_STATE_DICT['stop']  # end
			else:
				track_day_type = EDDY_STATE_DICT['continue']  # continue

			# adjust the PixelIdxList, because starts from 1 instead from 0
			espxl = eddy.Stats.PixelIdxList - 1
			eddy_num_pixels = espxl.shape[0]
			assert np.isfinite(eddy_num_pixels)

			skip_area_op = False

			# check if area was covered already
			num_overlap = np.count_nonzero(tidx.T.flat[espxl] > 0)
			if num_overlap > 0:
				# search for overlapping eddy
				d = tidx.T.flat[espxl]
				tid = d[d != 0][0]
				e = None
				for l in eddies:
					if l[3] == tid:
						e = l
						break
				# get size of the other eddy
				num_px_overlapping = np.count_nonzero(tidx == e[3])
				day_type_overlapping = e[6]
				
				print(date, ':: WARNING: detected overlap of {:.0f}px between track {:.0f} ({:.0f}) and track {:.0f} ({:.0f})'.format(num_overlap, track_id, eddy_type, e[3], e[4]))
				# if previous is larger than current, set current to fake
				if num_px_overlapping >= eddy_num_pixels:
					skip_area_op = True
					
					# this is a fix to capture the edge case, that we replace the very first eddy of a track
					# in this case the distance calculation does not find the eddy in the history object
					# with this it does and since it is the same eddy, the distance will be 0
					if track_day_type == EDDY_STATE_DICT['start']:
						history['last_eddy_index'][track_id] = eddy

					# set to fake eddy
					track_day_type = EDDY_STATE_DICT['fake']
					eddy_num_pixels = 0
					eddy_area = np.nan
					eddy_amplitude = np.nan
					eddy_in_subdomain = -1
					eddy_distance_coast = np.nan
					print(date, ':: INFO: converted eddy {:d} of track {:.0f} to fake eddy'.format(history['overall_eddy_counter'], track_id))
				# set previous to fake
				else:
					e[6] = EDDY_STATE_DICT['fake']  # track day type
					e[8] = 0  # num px
					e[9] = np.nan  # area
					e[10] = np.nan  # amplitude
					e[12] = -1 # in subdomain
					e[13] = np.nan # distance to coast
					tidx[tidx == e[3]] = 0
					eidx[eidx == e[0]] = 0

					print(date, ':: INFO: converted eddy {:d} of track {:.0f} to fake eddy'.format(e[0], e[3]))

			if not skip_area_op:
				assert (tidx.T.flat[espxl] == 0).all()
				assert (eidx.T.flat[espxl] == 0).all()

				# fill tidx and eidx
				tidx.T.flat[espxl] = track_id
				eidx.T.flat[espxl] = history['overall_eddy_counter']

				# calculate eddy amplitude
				bool_mask = eidx == history['overall_eddy_counter']
				assert np.count_nonzero(bool_mask) > 0
				bool_mask_edge = eddy_edge(bool_mask)
				mval = np.mean(zeta[bool_mask_edge])
				extremum = 0
				if eddy_type == EDDY_CYCL_DICT['cyclonic']:
					# cyclones are downwelling, so we are looking for the minimum
					extremum = np.nanmin(zeta[bool_mask])
				else:
					extremum = np.nanmax(zeta[bool_mask])
				
				eddy_amplitude = np.abs(extremum - mval)

				# get distance to coast
				eddy_distance_coast = np.nanmean(distance_map[bool_mask])

				# check if eddy in subdomain
				if (subdomain is not None) and np.count_nonzero(np.logical_and(subdomain, bool_mask)) > 0:
					eddy_in_subdomain = 1

				# area of eddy
				eddy_area = np.sum(area_map[bool_mask])

		# calculate the travelled distance (if not starting point)
		travelled_dist = 0
		if track_day_type != EDDY_STATE_DICT['start']:
			last_eddy = history['last_eddy_index'][track_id]
			last_lon, last_lat = last_eddy.Lon, last_eddy.Lat
			curr_lon, curr_lat = eddy.Lon, eddy.Lat
			travelled_dist = geodesic((last_lat, last_lon), (curr_lat, curr_lon)).km

		# save for coming fake eddies
		history['last_eddy_index'][track_id] = eddy

		# set full_eddy_id for later identification
		full_eddy_id = '{:d}-{:d}-{:d}'.format(time_index, eddy_type, eddy_idx)    
		# register eddy as from_track
		assert full_eddy_id not in eddies_from_track, (full_eddy_id, eddies_from_track)
		eddies_from_track.add(full_eddy_id)

		# age
		age = (date - history['eddy_start_times'][track_id]).total_seconds() / (3600 * 24)

		# save eddy data
		eddies.append([
			history['overall_eddy_counter'],
			eddy.Lat,
			eddy.Lon,
			track_id,
			eddy_type,
			age,
			track_day_type,
			lifetime,
			eddy_num_pixels,
			eddy_area,
			eddy_amplitude,
			travelled_dist,
			eddy_in_subdomain,
			eddy_distance_coast
		])
		eddy_dates.append(date)

	# loop over all eddies again to handle also those not assigned to any track
	skipped_eddies = 0
	for eddy_type, eddy_list in [(1, eddies_anti), (2, eddies_cycl)]:		
		for eddy_idx, eddy in enumerate(eddy_list):
			# create full_eddy_id and check if it was already used
			# NOTE: this works, because we have the same order (anti -> cycl)
			# for filling up the tracks, thats why eddy_idx matches indeed!
			full_eddy_id = '{:d}-{:d}-{:d}'.format(time_index, eddy_type, eddy_idx)
			if full_eddy_id in eddies_from_track:
				continue
			if full_eddy_id in eddies_not_from_track:
				raise RuntimeError
			# register eddy as not_from_track
			eddies_not_from_track.add(full_eddy_id)

			# increase counter
			history['overall_eddy_counter'] += 1
			# modify the PixelIdxList
			espxl = eddy.Stats.PixelIdxList - 1
			num_overlap = np.count_nonzero(tidx.T.flat[espxl] > 0)
			if num_overlap > 0:
				d = tidx.T.flat[espxl]
				tid = d[d != 0][0]
				e = None
				for l in eddies:
					if l[3] == tid:
						e = l
						break
				print(date, ':: WARNING: detected overlap of {:.0f}px between track {:.0f} ({:.0f}) and non-tracked eddy {:.0f} ({:.0f}). Non-tracked is dropped.'.format(num_overlap, e[3], e[4], history['overall_eddy_counter'], eddy_type))
				skipped_eddies += 1
				continue

			# assert (tidx.T.flat[espxl] == 0).all()
			# assert (eidx.T.flat[espxl] == 0).all()
			
			# save to map
			tidx.T.flat[espxl] = -2
			eidx.T.flat[espxl] = history['overall_eddy_counter']

			# calculate eddy amplitude (see above)
			bool_mask = eidx == history['overall_eddy_counter']
			bool_mask_edge = eddy_edge(bool_mask)
			zb = zeta[bool_mask].reshape(-1)
			extremum = zb[np.nanargmax(np.abs(zb))]
			eddy_amplitude = np.abs(extremum - np.nanmean(zeta[bool_mask_edge]))
			# area of eddy
			eddy_area = np.sum(area_map[bool_mask])
			# distance to coast
			eddy_distance_coast = np.nanmean(distance_map[bool_mask])

			eddy_in_subdomain = 0
			if (subdomain is not None) and np.count_nonzero(np.logical_and(subdomain, bool_mask)) > 0:
				eddy_in_subdomain = 1

			eddy_num_pixels = espxl.shape[0]
			assert np.isfinite(eddy_num_pixels)
				
			# save eddy data
			eddies.append([
				history['overall_eddy_counter'],
				eddy.Lat,
				eddy.Lon,
				-1,
				eddy_type,
				-1,
				EDDY_STATE_DICT['unknown'],
				0,
				eddy_num_pixels,
				eddy_area,
				eddy_amplitude,
				0,
				eddy_in_subdomain,
				eddy_distance_coast
			])
			# have to do it extra, because otherwise numpy array gets type object :/
			eddy_dates.append(date)

	assert len(eddies) + skipped_eddies == len(eddies_anti) + len(eddies_cycl) + fake_eddies
	
	# save locations of filaments
	if fillaments is not None:
		tidx[fillaments.T == 1] = -2

	# convert to numpy array
	eddies = np.expand_dims(np.array(eddies).T, 1)

	assert (eddies[3, :, :] != 0).all()

	return eddies, tidx, eidx, history


def write_to_shard(eddy_ds, tidx, eidx, in_shard_idx, date, eddies):
	"""
	Write eddy data to an output shard.
	"""
	# conversion function for date objects
	conv_fn = partial(date2num, units=eddy_ds.variables['time'].units, calendar=eddy_ds.variables['time'].calendar)
	# write data to output
	eddy_ds.variables['eidx'][in_shard_idx, :eddies.shape[2]] = eddies[0, :, :].astype(int)
	eddy_ds.variables['lat'][in_shard_idx, :eddies.shape[2]] = eddies[1, :, :].astype(float)
	eddy_ds.variables['lon'][in_shard_idx, :eddies.shape[2]] = eddies[2, :, :].astype(float)
	eddy_ds.variables['time'][in_shard_idx] = conv_fn(date)
	eddy_ds.variables['doy'][in_shard_idx] = date.dayofyr - 1
	eddy_ds.variables['tidx'][in_shard_idx, :eddies.shape[2]] = eddies[3, :, :].astype(int)
	eddy_ds.variables['cyc'][in_shard_idx, :eddies.shape[2]] = eddies[4, :, :].astype(int)
	eddy_ds.variables['age'][in_shard_idx, :eddies.shape[2]] = eddies[5, :, :].astype(int)
	eddy_ds.variables['state'][in_shard_idx, :eddies.shape[2]] = eddies[6, :, :].astype(int)
	eddy_ds.variables['lifetime'][in_shard_idx, :eddies.shape[2]] = eddies[7, :, :].astype(int)
	eddy_ds.variables['tidx_map'][in_shard_idx, :, :] = np.expand_dims(tidx.T, 0)
	eddy_ds.variables['eidx_map'][in_shard_idx, :, :] = np.expand_dims(eidx.T, 0)
	eddy_ds.variables['num_px'][in_shard_idx, :eddies.shape[2]] = eddies[8, :, :].astype(int)
	eddy_ds.variables['area'][in_shard_idx, :eddies.shape[2]] = eddies[9, :, :]
	eddy_ds.variables['amplitude'][in_shard_idx, :eddies.shape[2]] = eddies[10, :, :]
	eddy_ds.variables['distance'][in_shard_idx, :eddies.shape[2]] = eddies[11, :, :]
	eddy_ds.variables['in_subdomain'][in_shard_idx, :eddies.shape[2]] = eddies[12, :, :].astype(bool)
	eddy_ds.variables['distance_to_coast'][in_shard_idx, :eddies.shape[2]] = eddies[13, :, :]


def process_folder(base_path, output, num_items_shard=3000000, debug=False, base_folder='ssh'):
	"""
	Process a folder - this requires a special data structure in the folder
	"""

	history = {
		'used_track_day_idxs': defaultdict(set),  # keep track of all idxs inside a given track to make sure that we don't use them twice
		'last_eddy_index': {},  # track -> last recognized eddy, to deduce some stuff for fake eddies
		'eddy_start_times': {},  # track -> time of first eddy in track
		'overall_eddy_counter': 1
	}

	### Load data

	# input dataset
	ds = open_dataset(os.path.join(base_path, base_folder+'/ssh.nc'), ['time', 'zeta'])
	freq = int((ds.time.isel(time=1).values.item() - ds.time.isel(time=0).values.item()).total_seconds() / (3600 * 24))
	print('Frequency is', freq, 'days')

	# create a date-track-index
	date_track_idx, tracks = get_date_track_index(
		os.path.join(base_path, base_folder+'/tracks/joined_tol2day_anticycl_tracks.mat'),
		os.path.join(base_path, base_folder+'/tracks/joined_tol2day_cyclonic_tracks.mat'),
		ds, freq
	)

	# grid data
	grid_data = np.load(os.path.join(base_path, 'grid.npz'))
	subdomain_mask = grid_data['subdomain'].T
	distance_map = grid_data['distance_map'].T
	grid = xr.open_dataset(os.path.join(base_path, 'grid.nc'))
	area_map = get_area_map(grid).T

	# some paths
	eddy_paths = {
		'cycl': os.path.join(base_path, base_folder+'/eddies/cyclonic_{:d}.mat'),
		'anti': os.path.join(base_path, base_folder+'/eddies/anticyc_{:d}.mat')
	}
	raw_ssh_path = os.path.join(base_path, base_folder+'/raw/ssh_{:d}.mat')

	# shortcut for creating a new shard
	eddy_cycl_type = None 
	eddy_state_type = None
	eddy_ds = None
	shard_nr = None
	def new_shard():
		nonlocal eddy_cycl_type, eddy_state_type, eddy_ds, shard_nr
		eddy_ds, shard_nr, eddy_cycl_type, eddy_state_type = init_shard(output, ds, eddy_ds, shard_nr)

	# get times
	time_idxs = list(range(ds.dims['time']))
	in_shard_idx = 0

	# loop different times
	for i, time_idx in enumerate(time_idxs):

		# create initial shard and when maximum number of items per shard is reached
		if i % num_items_shard == 0:
			new_shard() 
			in_shard_idx = 0

		date = ds.time.isel(time=time_idx).values.item()
		print('Processing', date, end='\r')
		
		# run ssh debugging
		if debug:
			run_ssh_debug(ds, time_idx, raw_ssh_path, freq)

		# transform data to eddy data
		eddies, tidx, eidx, history = run_time_index(ds, time_idx, freq, area_map, distance_map, date_track_idx, tracks, eddy_paths, history, fillaments_ds=None, subdomain=subdomain_mask)
		write_to_shard(eddy_ds, tidx, eidx, in_shard_idx, date, eddies)
		in_shard_idx += 1

		# close last shard
		if i == len(time_idxs) - 1:
			eddy_ds.close()
			print('\nClosed last shard.')


if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--input", type=str, help="Input dataset", required=True)
	parser.add_argument("-o", "--output", type=str, help="Path to output folder", required=True)
	parser.add_argument("--debug", action='store_true', help="Enable debugging (i.e. SSH checks)")
	
	args = parser.parse_args()

	# transform folder (with special data structure) to eddy output
	base_path = '/nfs/kryo/work/maxsimon/data/{:s}/'.format(args.input)
	process_folder(base_path, args.output, debug=args.debug, base_folder='ssh')
