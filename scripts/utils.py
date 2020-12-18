#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


import os
import sys
import numpy as np
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator, griddata
from scipy.ndimage.filters import uniform_filter1d
import datetime
import netCDF4 as nc
import cftime


def get_area_map(grid_data, interpolate_to_psi=False):
	"""
	Calculate the area of grid cells
	"""
	pm = None
	pn = None
	if interpolate_to_psi:
		# TODO: use interp.py
		coords = np.vstack((grid_data.lon_rho.values.reshape(-1),
						   grid_data.lat_rho.values.reshape(-1))).T
		pm = LinearNDInterpolator(coords, grid_data.pm.values.reshape(-1)
								  )(grid_data.lon_psi.values, grid_data.lat_psi.values)
		pn = LinearNDInterpolator(coords, grid_data.pn.values.reshape(-1)
								  )(grid_data.lon_psi.values, grid_data.lat_psi.values)
	else:
		pm = grid_data.pm.values
		pn = grid_data.pn.values

	area = (1/pm) * (1/pn)
	return area / (1000.*1000.)


def parse_slice(val):
	"""
	Convert a string with a Python-like slice notation to a slice object.
	"""
	if ':' not in val:
		value = int(val)
		stop_value = value + 1 if value != -1 else None
		return slice(value, stop_value)
	else:
		value = val.split(':')
		start = None if value[0] == '' else int(value[0])
		stop = None if value[1] == '' else int(value[1])
		step = None if len(value) < 3 or value[2] == '' else int(value[2])
		return slice(start, stop, step)


def parse_datetime_string(date_string):
	"""
	Parse a string to a datetime object by checking different formats. Also returns the format.
	"""
	date = None
	date_f = None
	for date_format in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S.%f']:
		try:
			date = datetime.datetime.strptime(date_string, date_format)
			date_f = date_format
			break
		except ValueError:
			pass
	if date is None:
		raise ValueError('Could not find a suitable date format.')
	return date, date_f


def date_string_to_obj(date_string, sample_obj):
	"""
	Parse a string to an object given by sample_obj. The constructor must accept common datetime attributes (see code). This is especially useful when working with cftime.
	"""
	dt_obj, _ = parse_datetime_string(date_string)
	return type(sample_obj)(year=dt_obj.year, month=dt_obj.month, day=dt_obj.day, hour=dt_obj.hour, minute=dt_obj.minute, second=dt_obj.second)


def add_to_date_string(date_string, dt):
	"""
	Add a timedelta object to a date string.
	"""
	# parse the date string
	date_start, _ = parse_datetime_string(date_string)
	# format it correctly for xarray
	date_end = (date_start + dt).strftime('%Y-%m-%dT%H:%M:%S')
	# bugfix: strftime strips leading zeros
	first_idx = date_end.index('-')
	if first_idx != 4:
		date_end = '0'*(4 - first_idx) + date_end
	# if not time was provided in initial string, just return the date part
	if ':' in date_string:
		return date_end
	else:
		return date_end.split('T')[0]


def get_lon_lat_dims(dataarray):
	"""
	Get the name of lon and lat corresponding to an dataarray (based on the dimensions of the dataarray).
	"""
	# get correct grid
	dims = dataarray.dims
	lon_name = 'lon_rho'
	lat_name = 'lat_rho'
	for dim in dims:
		if dim.startswith('eta') or dim.startswith('lon'):
			lon_name = dim.replace('eta_', 'lon_')
		if dim.startswith('xi') or dim.startswith('lat'):
			lat_name = dim.replace('xi_', 'lat_')
	assert lon_name.replace('lon_', '') == lat_name.replace('lat_', ''), 'Ey, lon_rho != lon_u altough eta_rho == eta_u'
	return lon_name, lat_name


def get_depth_dim(dataarray):
	"""
	Filter the depth dimension of a data array.
	"""
	if 'depth' in dataarray.dims:
		return 'depth'
	if 's_rho' in dataarray.dims:
		return 's_rho'
	return None


def check_output_path(output_path):
	"""
	Check that a file does not exist yet at an output path and ask the user what to do if it exists.
	"""
	if os.path.isfile(output_path):
		print('WARNING: a file exist at the specified output path')
		action = input('Do you want to overwrite (o) or cancel (c)? ')

		if action.strip() == 'c':
			sys.exit()

		elif action.strip() == 'o':
			# do same as it would not exist
			pass

		else:
			print('ERROR: unknown option.')
			sys.exit(1)


def get_num_days(dataset):
	"""
	Parse the time:calendar attribute of a dataset and get the number of days a year has
	"""
	if "time" in dataset:
		# get the max days from calendar
		calendar = dataset["time"].attrs['calendar']
		max_days = int(calendar.replace("_day", ""))
		return max_days
	else:
		return len(dataset["doy"])


def get_doys(t_obj, ds, days_around):
	"""
	Get an array of all doys which are `days_around` days around a time object.
	"""
	doy = t_obj.dayofyr - 1
	num_days = get_num_days(ds)
	doys = np.array([i % num_days for i in range(doy - days_around, doy + days_around + 1)])
	assert len(doys) == days_around*2 + 1
	return doys


def get_doys_around_doy(doy, num_days, days_around):
	"""
	Get an array of all doys which are `days_around` days around a doy.
	"""
	doys = np.array([i % num_days for i in range(doy - days_around, doy + days_around + 1)])
	assert len(doys) == days_around*2 + 1
	return doys


def str_tobj(t_obj):
	"""
	Pretty pring a time object (cftime)
	"""
	if type(t_obj) is not cftime._cftime.DatetimeNoLeap:
		return str(t_obj)
	else:
		return '{:04d}-{:02d}-{:02d}'.format(t_obj.year, t_obj.month, t_obj.day)


def get_triangular_weights(doys):
	"""
	Get an array of weights for triangular weighting.
	"""
	weights = np.zeros(doys.shape[0]).astype(float)
	width = doys.shape[0] // 2 + 1
	half_weights = np.linspace(0, 1, width)
	if doys.shape[0] % 2 == 0:
		weights[:width-1] = half_weights[:-1]
		weights[width-1:] = half_weights[::-1][1:]
	else:
		weights[:width] = half_weights
		weights[width:] = half_weights[::-1][1:]
	return weights / np.sum(weights)


def np_rolling_mean(data, num_points, mode="reflect", axis=0):
	"""
	Calculating a rolling mean on a numpy array.
	"""
	return uniform_filter1d(data, size=num_points, axis=axis, mode=mode)


def p(fn, *args, **kwargs):
	"""
	Get a callable which - when executed - executes a function with given arguments and keyword-arguments.
	This is used in the context of `cache`.
	"""
	def s():
		return fn(*args, **kwargs)
	return s


def cache(path, *args, invalidate=False):
	"""
	Cache the result of a list of callables. The callables are only executed when the provided path does not exist. 
	"""
	data = None
	args_keys = ['{:d}'.format(i) for i in range(len(args))]
	# load cache
	if os.path.isfile(path+'.npz') and not invalidate:
		print('Load cache')
		data = np.load(path+'.npz', allow_pickle=True)
	# execute all callables and save results to numpy
	else:
		data = {
			args_keys[i]: args[i]() for i in range(len(args))
		}
		np.savez(path+'.npz', **data)
	return [data[key] for key in args_keys]


def mean_with_error(x, dx, axis=None):
	"""
	Calculate an average and propagate the error accordingly.
	"""
	# calculate mean: f(x) = 1/N * (x1 + x2 + ...)
	mean = np.nanmean(x, axis=axis)
	num_nonnan = np.count_nonzero(~np.isnan(dx), axis=axis)
	# error propagation: df(x) = 1/N * sqrt(dx1**2 + dx2**2 + ...)
	dk = np.sqrt(
		np.nansum(dx**2, axis=axis)
	) / num_nonnan
	return mean, dk


def ratio_with_error(x, y, dx, dy):
	"""
	Calculate a ratio and propagate the errors accordingly.
	"""
	# f(x, y) = x/y
	rel = x/y
	# df(x, y) = sqrt( (dx/y)**2 + (dy*x/(y**2))**2 )
	d_rel = np.sqrt(
		((dx/y)**2) +
		((dy*x/(y**2))**2)
	)
	return rel, d_rel
