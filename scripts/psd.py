#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


import sys
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')

import numpy as np
import scipy.signal as sig
from romstools.utils import str_tobj

# labels for yaxis when using density for different variables
PSD_LABELS_DENSITY = {
	'u': "KE $\quad [m^5 s^{-4}]$",
	'v': "KE $\quad [m^5 s^{-4}]$",
	'w': "KE $\quad [m^5 s^{-4}]$",
	'u+iv': "KE $\quad [m^5 s^{-4}]$",
	'u+v': "KE $\quad [m^5 s^{-4}]$",
	'v+u': "KE $\quad [m^5 s^{-4}]$",
	'u_b+v_b': "KE $\quad [m^5 s^{-4}]$",
	'v_b+u_b': "KE $\quad [m^5 s^{-4}]$",
	'temp': "Temp $\quad [K^2 m]$",
	'rvort': 'TODO',
	'TOT_PROD': 'NPP $\quad$ [mol / m$^2$ / yr]',
	'zeta': 'SSH $\quad$ [m^3]'
}

# labels for yaxis when using spectrum for different variables
PSD_LABELS_SPECTRUM = {
	'u': "KE $\quad [m^4 s^{-4}]$",
	'v': "KE $\quad [m^5 s^{-4}]$",
	'w': "KE $\quad [m^5 s^{-4}]$",
	'u+iv': "KE $\quad [m^4 s^{-4}]$",
	'u+v': "KE $\quad [m^4 s^{-4}]$",
	'v+u': "KE $\quad [m^4 s^{-4}]$",
	'temp': "Temp $\quad [K^2]$",
	'rvort': 'TODO',
	'TOT_PROD': 'NPP $\quad$ [mol / m$^3$ / yr]',
	'zeta': 'SSH $\quad$ [m^2]'
}

# label for xaxis for temporal (f) and spatial (k) frequencies
PSD_XLABEL = {
	'k': '$k\quad[m^{-1}]$',
	'f': '$f\quad[s^{-1}]$'
}


def get_dist(x, axis):
	"""
	Calculate 1/x and return full range of data. The range of data is given as x < means < y with x = means - d_min and y = means + d_max.
	Range is d_min, d_max
	"""
	means = np.mean(x, axis=axis)
	means = 1 / means
	means_range = np.array([
		means - np.nanmin(1/x, axis=axis),
		np.nanmax(1/x, axis=axis) - means
	])
	return means, means_range


def binned_mean(x, y, num):
	"""
	Create a binned mean of x and y. This reduces a lot of noise in the PSD
	"""
	l = np.logical_and(np.isfinite(x), x > 0)
	# need to do it in logspace as the PSD will be in logspace as well.
	levels = np.logspace(np.log10(np.nanmin(x[l])), np.log10(np.nanmax(x[l])), num)
	x_ = []
	y_ = []
	yerr_ = []
	# loop levels
	for i in range(1, len(levels)):
		# get all items belonging to the bin
		m = np.logical_and(x >= levels[i-1], x < levels[i])
		if np.count_nonzero(m) > 0:
			# ... and use averages of the bin
			x_.append(np.nanmean(x[m]))
			y_.append(np.nanmean(y[m]))
			yerr_.append(np.nanstd(y[m]))
		else:
			# just add 0s
			x_.append(0)
			y_.append(0)
			yerr_.append(0)
	return np.array(x_), np.array(y_), np.array(yerr_)


def do_multidim_welch(data, grid_distances, axis, aggregation_mode='mean', scaling="density"):
	"""
	Do a multidimensional welch. The data is 2d. This method calculates seperate PSDs of data along axis and aggregates them to a single PSD.
	grid_distances sets the distance for each PSD. aggregation_mode can be 'mean' (for each entry take the mean of all PSDs), 'bin' (sort data, create 100 bins and take the mean for each bin) and 'none' (just sort the data).
	"""
	x = []
	y = []
	yerr = []
	
	# define the complementary axis to axis
	compl_axis = (axis + 1)%2
	assert data.shape[compl_axis] == len(grid_distances), (data.shape, grid_distances.shape)
	
	# for each line do a seperate welch
	# because the mean grid distance migt be different
	# to avoid this loop, just take
	# x, y = sig.welch(data, 1/np.mean(grid_distances), scaling=scaling, axis=axis)
	# and flip the axis accordingly
	for i in range(len(grid_distances)):
		d = data[i] if axis == 1 else data[:, i]
		x_, y_ = sig.welch(d, 1/grid_distances[i], scaling=scaling)
		x.append(x_)
		y.append(y_)
		
	x = np.array(x)
	y = np.array(y)
	
	if aggregation_mode == 'mean':
		# aggregate data by taking the mean
		x = np.mean(x, axis=0)
		yerr = np.array([
			np.nanmin(y, axis=0),
			np.nanmax(y, axis=0)
		])
		y = np.mean(y, axis=0)
		
	elif aggregation_mode == 'bin':
		## flatten
		x = x.reshape(-1)
		y = y.reshape(-1)
		## sort data
		sort = np.argsort(x)
		x = x[sort]
		y = y[sort]
		## do a binned mean, because otherwise very noisy
		x, y, yerr = binned_mean(x, y, 50)
		
	else:
		# do not aggregate, just put everything together
		x = x.reshape(-1)
		y = y.reshape(-1)
		s = np.argsort(x)
		x = x[s]
		y = y[s]
		yerr = np.array([0 for _ in x])
		
	# test shapes
	assert x.shape == y.shape
	return x, y, yerr


def prepare_axis_for_psd(ax, var_name=None, xlabel=None, scaling=None, vlines=None, pows=None, xlim=None, ylim=None):
	"""
	Adjust figure for PSD plot. I.e. set the labels, limits and scales. Also vertical lines as well as lines of constant power (appearing as linear lines in plot) can be plotted.
	"""
	
	ls = ['-', '-.', '--', ':']
	# set scale and limits
	ax.set_xscale('log')
	ax.set_yscale('log')
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)

	# set labels
	if xlabel is not None:
		ax.set_xlabel(PSD_XLABEL[xlabel])

	if scaling is not None and var_name is not None:
		ylabel = PSD_LABELS_DENSITY[var_name] if scaling == 'density' else PSD_LABELS_SPECTRUM[var_name]
		ax.set_ylabel(ylabel)

	# add vertical lines
	if vlines is not None:
		for vline in vlines:
			ax.axvline(vline, 0, 1, color='k')
	
	# add power laws
	if pows is not None:
		pow_x = pows['x']
		pow_y = pows['y']
		for pow_i, pow in enumerate(pows['pows']):
			y = pow_x ** pow
			# scale such that all cross pow_y
			loc = y[len(pow_x) // 2] / pow_y
			y /= loc
			ax.plot(pow_x, y, label="k$^{"+"{:1.1f}".format(pow)+"}$", color='grey', ls=ls[pow_i])


def plot_single_psd(ax, t, x, y, d_y, label, color, with_error=False):
	"""
	Add PSD data to a figure
	"""
	mask = x > 0
	# plot psd to axis
	l = []
	# add plots
	l.append(ax.plot(x[mask], y[mask], marker='.', ls="-", c=color, label=label))
	# plot error range
	if with_error and np.sum(d_y) > 0:
		dymin = d_y[mask] if len(d_y.shape) == 1 else d_y[0][mask]
		dymax = d_y[mask] if len(d_y.shape) == 1 else d_y[1][mask]
		l.append(ax.fill_between(x[mask], y[mask] - dymin, y[mask] + dymax, color=color, alpha=0.3))
	# set title
	if type(t) == list or type(t) == np.ndarray:
		if len(t) > 1:
			ax.set_title(str_tobj(t[0]) + ' - ' + str_tobj(t[-1]))
		else:
			ax.set_title(str_tobj(t[0]))
	else:
		ax.set_title(str_tobj(t))
		
	return l


def do_complex_time_welch(run, frequency, depth_slice=None, eta_slice=slice(None, None), xi_slice=slice(None, None), time_slice=None):
	"""
	Calculate a complex PSD of u and v. This allows to seperate between cyclonic and anticyclonic motions.
	"""
	# get dimensions
	num_times, num_depths, num_eta, num_xi = run.u.shape
	depth_dim = 'depth' if 'depth' in run.u.dims else 's_rho'
	
	# get possible eta idxs
	eta_idxs = np.arange(num_eta)[eta_slice]
	# get possible depths
	depth_idxs = np.arange(num_depths)
	if depth_slice is None:
		ds = slice(0, 1) if depth_dim == 'depth' else slice(-1, 0)
		depth_idxs = depth_idxs[ds]
	else:
		depth_idxs = depth_idxs[depth_slice]
	
	# lists which are used to average at the end
	all_x = None
	all_y = []
	all_d_y = []

	# loop depths
	for depth_idx in depth_idxs:
		# create dictionary to select data
		sel = {
			'eta_rho': eta_slice,
			'xi_rho': xi_slice,
			depth_dim: depth_idx
		}
		if time_slice is not None:
			sel['time'] = time_slice

		# get data. set v as imaginary part
		var = 1j * np.squeeze(run.v.isel(**sel).values.copy())
		var += np.squeeze(run.u.isel(**sel).values.copy())
		var = var.reshape(var.shape[0], -1)

		# do welch calculation
		x, y = sig.welch(var, frequency, axis=0, scaling="density")
		y_err = np.nanstd(y, axis=1)
		y = np.nanmean(y, axis=1)
		x = np.squeeze(x)
		y = np.squeeze(y)
		
		# some consistency tests
		if all_x is not None:
			assert (x == all_x).all()
		else:
			all_x = x
		
		# append result
		all_y.append(y)
		all_d_y.append(y_err)

	# return averages
	return all_x, np.nanmean(all_y, axis=0), np.nanmean(all_d_y, axis=0)
