#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


import sys
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')

import xarray as xr
import numpy as np
import os
from romstools.dataset import open_dataset
from romstools.slice import slice_on_rho_grid
from functools import partial
from romstools.utils import get_area_map, date_string_to_obj, get_lon_lat_dims, get_depth_dim
from romstools.plot import plot_data
from romstools.psd import get_dist, do_multidim_welch
from datetime import timedelta as tdelta
import progressbar
import warnings
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import cartopy.crs as ccrs



class RomsRun:

	def __init__(self, grid_path, eta_slice=None, xi_slice=None, s_rho_slice=None):
		"""
		Get a RomsRun object which can handle different netCDF datasets belonging to the same run. 
		"""

		# open grid data and slice
		self.grid_path = grid_path
		self.grid = xr.open_dataset(self.grid_path)
		self.grid = slice_on_rho_grid(self.grid, eta_rho_slice=eta_slice, xi_rho_slice=xi_slice, s_rho_slice=s_rho_slice)

		# calculate area map
		self.area_map = get_area_map(self.grid)

		# save slices
		self._eta_slice = eta_slice
		self._xi_slice = xi_slice
		self._s_rho_slice = s_rho_slice

		# initialize internal data
		self._data = {}
		self._reverse_data = {}
		self._reverse_prefix = {}
		self._prefixes = {}
		self._psd_slices = {}
		self._depth_key = None
		self._depth_num = None
		self.time = None


	def add_data(self, path, variables=[], time_calendar=None, time_raw=None, time_units=None, var_prefix='', time_offset=0, time_from=None):
		"""
		Add a netCDF dataset to the object.
		"""
		
		if path in self._data:
			print('WARNING: this path has been loaded already')
			return

		# open dataset
		data = open_dataset(path, variables, time_calendar=time_calendar, time_raw=time_raw, time_offset=time_offset, time_from=time_from, time_units=time_units, eta_rho_slice=self._eta_slice, xi_rho_slice=self._xi_slice, s_rho_slice=self._s_rho_slice)
		depth_key = get_depth_dim(data)
		
		# do some consistency tests...

		# ... check depth
		assert self._depth_key is None or depth_key is None or depth_key == self._depth_key
		assert self._depth_num is None or depth_key is None or data.dims[depth_key] == self._depth_num

		# ... check if time is the same
		if 'time' in data:
			assert self.time is None or \
				   (len(self.time) == len(data['time']) and \
				    abs((self.time[0] - data['time'][0]).values) < np.timedelta64(1, 'h') and \
					abs((self.time[-1] - data['time'][-1]).values) < np.timedelta64(1, 'h')), (self.time[0], data.time[0], self.time[-1], data.time[-1])
			self.time = data.time

		# save depth key (s_rho vs depth)
		if depth_key is not None:
			self._depth_key = depth_key
			self._depth_num = data.dims[depth_key]

		if depth_key == 'depth' and 'depth' in data.coords:
			setattr(self, 'depth', data.depth)
		
		# save full object for direct access
		self._data[path] = data
		self._prefixes[path] = var_prefix
		
		# save each single variables
		for var_name in data:
			if getattr(self, var_prefix + var_name, None) is not None:
				print('WARNING:', var_prefix + var_name, 'already exist. Consider to change `var_prefix`.')
				continue
			setattr(self, var_prefix + var_name, data[var_name])
			self._reverse_data[var_prefix + var_name] = path
			self._reverse_prefix[var_prefix + var_name] = var_prefix


	def close(self):
		"""
		Close each dataset associated to the object
		"""
		self._data.close()
		for ds in self._data.values():
			ds.close()
		

	def slice_on_rho_grid(self, eta_rho_slice=None, xi_rho_slice=None):
		"""
		Slice all datasets associated to the object
		"""
		# TODO: this is somehow the same as doing this with the initializor.
		self.grid = slice_on_rho_grid(self.grid, eta_rho_slice=eta_rho_slice, xi_rho_slice=xi_rho_slice)
		for key, ds in self._data.items():
			self._data[key] = slice_on_rho_grid(ds, eta_rho_slice=eta_rho_slice, xi_rho_slice=xi_rho_slice)
			prefix = self._prefixes[key]
			for var_name in self._data[key]:
				# check if attribute exists
				getattr(self, prefix + var_name)
				setattr(self, prefix + var_name, self._data[key][var_name])


	def get_iterator(self, dim, func_or_var, value_slice=None):
		"""
		Get an iterator object (over time or depth) for a variable (str) or a function (callable) for which the argument t/depth_idx is filled with the iterator value.
		"""
		if dim == 'time':
			# get a list of time object and loop
			for t in self.get_time_list(value_slice):
				# if callable, fill t with object
				if callable(func_or_var):
					yield t, partial(func_or_var, t=t)
				# just return time object
				elif func_or_var is None:
					yield t, None
				# return attribut from object
				else:
					yield t, getattr(self, func_or_var).sel(time=t, method='nearest')
		elif dim == 'depth':
			# get a range for all depth items
			range_obj = range(self._depth_num) if self._depth_key == 'depth' else list(range(self._depth_num))[::-1]
			if value_slice is not None:
				range_obj = range_obj[value_slice]
			# loop depths
			for depth_idx in range_obj:
				# if callable, fill depth_idx with object
				if callable(func_or_var):
					yield depth_idx, partial(func_or_var, depth_idx=depth_idx)
				# just return depth
				elif func_or_var is None:
					yield depth_idx, None
				# return attribut from object
				else:
					yield depth_idx, getattr(self, func_or_var).isel(**{self._depth_key: depth_idx})
		else:
			raise RuntimeError('Unsupported dimension')


	def get_time_obj(self, t):
		"""
		Convert an integer, string or date object to a valid time object. If t is None, return the first timestamp in data.
		"""
		# get a valid time object	
		t_obj = None
		if getattr(self, 'time', None) is None:
			return None
		if t is None:
			t_obj = self.time[0]
		elif type(t) is int:
			t_obj = self.time[t]
		elif type(t) is str:
			t_obj = date_string_to_obj(t, self.time[0].item())
		else:
			t_obj = t
		return t_obj


	def get_time_list(self, time_slice):
		"""
		Create a list of possible time objects
		"""
		if type(time_slice) == slice and (type(time_slice.start) == str or type(time_slice.stop) == str):
			return self.time.sel(time=time_slice).values
		elif type(time_slice) == slice and (type(time_slice.start) == int or type(time_slice.stop) == int):
			return self.time.isel(time=time_slice).values
		else:
			return self.time.values


	def plot(self, var_name, t=None, depth_idx=None, **kwargs):
		"""
		Plot 2D data of curvilinear grid. See `plot.plot_data` for details.
		"""
		title = var_name

		# get a valid time object
		t_obj = self.get_time_obj(t)

		# extract data
		data = getattr(self, var_name)

		# handle depth
		depth_key = get_depth_dim(data)
		if depth_key is not None:
			my_depth_idx = 0 if depth_key == 'depth' else -1
			my_depth_idx = my_depth_idx if depth_idx is None else depth_idx
			data = data.isel(**{depth_key: my_depth_idx})
			title += ', depth {:d}'.format(my_depth_idx)

		# handle time
		if 'time' in data.dims:
			data = data.sel(time=t_obj, method='nearest')
			real_time = self.time.sel(time=t_obj, method='nearest').values.item()
			title += ', time {}'.format(real_time)

		# handle doy
		if 'doy' in data.dims:
			my_doy = 0 if t is None else t
			data = data.isel(doy=my_doy)
			title += ', doy {:d}'.format(my_doy)

		return plot_data(self.grid, data, title=title, **kwargs)


	def animate(self, var_name, time_slice=None, **kwargs):
		"""
		Create animation of 2D data over time. See `self.plot` for details.
		"""
		as_contourfill = 'as_contourfill' in kwargs and kwargs['as_contourfill']
		figsize = (8, 4) if 'figsize' not in kwargs else kwargs['figsize']

		subplot_kw = {'projection': ccrs.PlateCarree()} if as_contourfill else None
		fig, ax = plt.subplots(1, 1, subplot_kw=subplot_kw, figsize=figsize)

		first_frame = True
		plots = []

		# TODO: just use ax.clear()

		def remove_or_remove_collection(item):
			if getattr(item, 'collections', None) is not None:
				return list([pcontour.remove() for pcontour in item.collections])
			else:
				return item.remove()

		def a(t_obj):
			nonlocal plots, first_frame
			list([remove_or_remove_collection(plot) for plot in plots])
			_, plots = self.plot(var_name, t_obj, ax=ax, is_animation_frame=not first_frame, **kwargs)
			first_frame = False
			print('Plotting', t_obj, end='\r')

		frames = self.get_time_list(time_slice)

		print('Create animation from {} to {}'.format(frames[0], frames[-1]))

		rc('animation', html='html5')
		anim = animation.FuncAnimation(fig, a, frames=frames, interval=100, blit=False)
		return anim


	def set_psd_slice(self, subdomain_slice, dim, skip_assert=False):
		"""
		Set the dimension used for calculating PSDs. This is an extra step to check that the error introduced by curvilinear
		grids is less than 5%. Choose the subdomain such that the error keeps low.
		"""
		assert dim in ['eta', 'xi'], 'Unknown dim'

		mask_rho = self.grid.mask_rho.values.copy()[subdomain_slice]
		assert np.count_nonzero(np.isnan(mask_rho)) == 0, 'There are nans in your subdomain'

		# get the correct grid distancing data
		gval = self.grid.pm.values.copy() if dim == 'eta' else self.grid.pn.values.copy()
		# slice for subdomain
		if subdomain_slice is not None:
			gval = gval[subdomain_slice]
		# calculate the distance in grid
		axis = 1 if dim == 'eta' else 0
		const_dim_dist, dist_range = get_dist(gval, axis=axis)

		# check dimensions
		assert len(const_dim_dist) == subdomain_slice[(axis + 1)%2].stop - subdomain_slice[(axis + 1)%2].start, 'Dimension Mismatch!'
		
		# check that relative error is below 5%
		max_rel_error = np.max((dist_range[0] + dist_range[1])/const_dim_dist)
		assert skip_assert or max_rel_error < 0.05, 'Relative error is too large, choose smaller subdomain'

		print('PSD Slice for', dim)
		print('\tMaximum relative error: {:2.3f}%'.format(max_rel_error * 100))
		print('\tGrid distance range: {:.1f}m - {:.1f}m'.format(np.min(const_dim_dist), np.max(const_dim_dist)))

		self._psd_slices[dim] = {
			'subdomain': subdomain_slice,
			'const_dim_dist': const_dim_dist,
			'axis': axis
		}


	def psd(self, var_name, t, depth_idx=None, aggregation_mode='mean', scaling='density', dim='eta', min_doy=0, max_doy=365):
		"""
		Calculate a PSD on the data. When having multiple depth levels, they are averaged. You can combine variables by putting a + in between (this is
		for example useful for horizontal velocities). See `psd.do_multidim_welch` for details.
		"""
		# handle lists of depth idxs:
		# calculate psd for each depth and average y and d_y
		if type(depth_idx) == list or type(depth_idx) == np.ndarray:
			all_y = []
			all_d_y = []
			all_t = None
			all_x = None
			for d_idx in depth_idx:
				all_t, all_x, y, d_y = self.psd(var_name, t, depth_idx=d_idx, aggregation_mode=aggregation_mode, scaling=scaling, dim=dim, min_doy=min_doy, max_doy=max_doy)
				all_y.append(y)
				all_d_y.append(d_y)
			return all_t, all_x, np.nanmean(all_y, axis=0), np.nanmean(all_d_y, axis=0)

		# handle u+v/v+u:
		# calculate psd for u and v seperately and just add them together
		if '+' in var_name:
			dg_time = None
			dg_x = None
			dg_y = None
			dg_dy = None
			vars = var_name.split('+')
			for var in vars:
				d = self.psd(var, t, depth_idx=depth_idx, aggregation_mode=aggregation_mode, scaling=scaling, dim=dim, min_doy=min_doy, max_doy=max_doy)
				# is first var
				if dg_x is None:
					dg_time = d[0]
					dg_x = d[1]
					dg_y = d[2]
					dg_dy = d[3] ** 2
				else:
					assert (d[1] == dg_x).all()
					dg_y += d[2]
					dg_dy += d[3] ** 2
			return dg_time, dg_x, dg_y, np.sqrt(dg_dy)
		
		# set slices first
		if dim not in self._psd_slices:
			raise RuntimeError('You need to set a subdomain using set_psd_slice for this dim first')

		# get subdomain info
		subdomain_slice = self._psd_slices[dim]['subdomain']
		const_dim_dist = self._psd_slices[dim]['const_dim_dist']
		axis = self._psd_slices[dim]['axis']

		list_real_t = []
		list_y = []
		list_d_y = []
		prev_x = None

		# if t is not a list of t_obj/t_idxs, create a list with length 1
		t_list = t
		if type(t) != list and type(t) != np.ndarray:
			t_list = [t]
		
		# loop over times and average results
		for t_el in t_list:

			data = getattr(self, var_name)

			# get data by t_obj
			if 'time' in data.dims:
				t_obj = self.get_time_obj(t_el)
				doy = t_obj.dayofyr - 1
				if doy < min_doy or doy >= max_doy:
					continue
				data = data.sel(time=t_obj, method='nearest')
				real_t = self.time.sel(time=t_obj, method='nearest').values.item()
			# get data by doy
			if 'doy' in data.dims:
				data = data.isel(doy=t_el)
				real_t = t_el

			# get correct depth
			depth_key = get_depth_dim(data)
			if depth_key is not None:
				my_depth_idx = 0 if depth_key == 'depth' else -1
				my_depth_idx = my_depth_idx if depth_idx is None else depth_idx
				data = data.isel(**{depth_key: my_depth_idx})

			# extract the variable values and slice to subdomain
			var = np.squeeze(data.values.copy())
			var = var[subdomain_slice]

			# do PSD
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", message="nperseg = 256 is greater")
				warnings.filterwarnings("ignore", message="divide by zero")
				x, y, d_y = do_multidim_welch(var, const_dim_dist, axis, aggregation_mode, scaling)

			# check that the x values are actually the same
			if prev_x is not None:
				assert (prev_x == x).all()
			else:
				prev_x = x

			# save data
			list_real_t.append(real_t)
			list_y.append(y)
			list_d_y.append(d_y)

		return list_real_t, prev_x, np.nanmean(list_y, axis=0), np.nanmean(list_d_y, axis=0)


	def __getitem__(self, key):
		"""
		Access attributes also by []
		"""
		return getattr(self, key)




