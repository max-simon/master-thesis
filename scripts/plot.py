#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


import sys
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
from romstools.utils import get_lon_lat_dims
from matplotlib.cm import get_cmap

def get_domain_mask(lon, lat, domain):
	"""
	Create slices for array that represents a given domain. The domain is in cartopy format (dictionary with N, S, W and E).
	"""
	mask = np.ones_like(lon).astype(bool)
	mask[:, :] = True
	if 'E' in domain:
		mask[lon < domain['E']] = False
	if 'W' in domain:
		mask[lon > domain['W']] = False
	if 'N' in domain:
		mask[lat > domain['N']] = False
	if 'S' in domain:
		mask[lat < domain['S']] = False

	axis_0 = np.sum(mask, axis=1) > 0
	axis_1 = np.sum(mask, axis=0) > 0
	
	idx_0_0 = np.argmax(axis_0)
	idx_0_1 = len(axis_0) - np.argmax(axis_0[::-1])
	idx_1_0 = np.argmax(axis_1)
	idx_1_1 = len(axis_1) - np.argmax(axis_1[::-1])

	sd_slices = (slice(idx_0_0, idx_0_1), slice(idx_1_0, idx_1_1))
	assert np.sum(mask[:idx_0_0]) == 0
	assert idx_0_1 == mask.shape[0] or np.sum(mask[idx_0_1:]) == 0
	assert np.sum(mask[:, :idx_1_0]) == 0
	assert idx_1_1 == mask.shape[1] or np.sum(mask[:, idx_1_1:]) == 0

	return sd_slices, mask[sd_slices]


def plot_data(grid_data, data, title='', ax=None, highlight_subdomain=None, lon_name=None, lat_name=None, highlight_subdomain_alpha=0.4, figsize=(5, 3), as_contourfill=False, vmin=None, vmax=None, domain=None, grid='single', second_grid_labels=False, cmap=None, cmap_subd=None, is_animation_frame=False, cbar_label='', levels=50, contours_levels=0, contours_color='black', alpha=1.0, colorbar=True, cbar_ticks=None, land_gray=False):
	"""
	Plot two-dimensional data on a curvilinear grid. See Code for all the different options
	"""
	plots = []

	# if axis was not provided, create one and make sure we call call_show at the end
	call_show = False
	if ax is None:
		subplot_kw = {'projection': ccrs.PlateCarree()} if as_contourfill else None
		fig, ax = plt.subplots(1, 1, subplot_kw=subplot_kw, figsize=figsize)
		call_show = True

	# get lon and lat
	if lon_name is None or lat_name is None:
		lon_name, lat_name = get_lon_lat_dims(data)
	lon = grid_data[lon_name].values.copy()
	lat = grid_data[lat_name].values.copy()

	# convert to numpy
	if type(data) == xr.DataArray:
		data = data.values
	
	# get data
	data_main = np.squeeze(data.copy())
	data_subd = None
	alpha_main = alpha
	etas = np.ones_like(lon) * np.arange(lon.shape[0])[:, np.newaxis]
	xis = np.ones_like(lon) * np.arange(lon.shape[1])[np.newaxis, :]
	
	# highlight a subdomain
	if highlight_subdomain is not None:
		data_subd = np.zeros_like(data_main) * np.nan
		data_subd[highlight_subdomain] = data_main[highlight_subdomain]
		alpha_main = highlight_subdomain_alpha
	
	# slice to domain
	if type(domain) is dict:  # {N: int, S: int, E: int, W: int}
		slices, domain_mask = get_domain_mask(lon, lat, domain)
		data_main = data_main[slices]
		data_main[~domain_mask] = np.nan  # nans are transparent in matplotlib
		if data_subd is not None:
			data_subd = data_subd[slices]
			data_subd[~domain_mask] = np.nan
		lon = lon[slices]
		lat = lat[slices]
		etas = etas[slices]
		xis = xis[slices]
	elif type(domain) is tuple:  # (slice-eta, slice-xi)
		data_main = data_main[domain]
		if data_subd is not None:
			data_subd = data_subd[domain]
		lon = lon[domain]
		lat = lat[domain]
		etas = etas[domain]
		xis = xis[domain]
	else:
		pass

	# sanitize vmin, vmax
	d = data_main if data_subd is None else data_subd
	vmin = vmin if vmin is not None else np.nanmin(d)
	vmax = vmax if vmax is not None else np.nanmax(d)

	# colorbar
	cax = None
	# get correct color schema
	cmap_subd = cmap if cmap_subd is None else cmap_subd
	
	if as_contourfill:  # contour fill --> curvilinear to real data
		# plot data
		data_main[data_main > vmax] = vmax
		data_main[data_main < vmin] = vmin
		plots.append(ax.contourf(lon, lat, data_main, levels=levels, alpha=alpha_main, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax))
		# plot subdomain
		if highlight_subdomain is not None:
			data_subd[data_subd > vmax] = vmax
			data_subd[data_subd < vmin] = vmin
			plots.append(
				ax.contourf(lon, lat, data_subd, levels=levels, transform=ccrs.PlateCarree(), cmap=cmap_subd, vmin=vmin, vmax=vmax, alpha=alpha)
			)
		cax = plots[-1]
		
		# add contour lines
		if contours_levels > 0:
			plots.append(
				ax.contour(lon, lat, data_main if highlight_subdomain is None else data_subd, levels=contours_levels, transform=ccrs.PlateCarree(), colors=contours_color)
			)

		# add some other stuff which should be added only once in an animation
		if not is_animation_frame:

			ax.coastlines()
			ax.set_xlabel('Lon')
			ax.set_ylabel('Lat')

			## grids
			# plot lon-lat grid
			if grid == 'single' or grid == 'both' or grid == 'lonlat':
				gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree())
				gl.xlabels_top = False
				gl.ylabels_right = False
			# plot rho grid
			if grid == 'both' or grid == 'rho':
				xi_levels = np.arange(np.min(xis), np.max(xis))
				xi_levels = xi_levels[::len(xi_levels) // 10]
				eta_levels = np.arange(np.min(etas), np.max(etas))
				eta_levels = eta_levels[::len(eta_levels) // 10]
				cax_eta = ax.contour(lon, lat, etas, eta_levels, transform=ccrs.PlateCarree(), colors='grey', linewidths=1)
				cax_xi = ax.contour(lon, lat, xis, xi_levels, transform=ccrs.PlateCarree(), colors='grey', linewidths=1)
				if second_grid_labels:
					ax.clabel(cax_eta, fmt="%1.0f")
					ax.clabel(cax_xi, fmt="%1.0f")

	else:  # plot as image data
		plots.append(ax.imshow(np.rot90(np.flip(data_main, axis=1), k=1), alpha=alpha_main, vmin=vmin, vmax=vmax, cmap=cmap))
		if highlight_subdomain is not None:
			plots.append(ax.imshow(np.rot90(np.flip(data_subd, axis=1), k=1), alpha=alpha, vmin=vmin, vmax=vmax, cmap=cmap_subd))
		cax = plots[-1]

		# add contour lines
		if contours_levels > 0:
			# TODO: dont need transformation as like np.rot90(np.flip(XXX, axis=1))? Because of etas and xis?
			plots.append(
				ax.contour(etas, xis, data_main if highlight_subdomain is None else data_subd, levels=contours_levels, colors=contours_color)
			)
		
		# add some other stuff which should be added only once in an animation
		if not is_animation_frame:
			
			# labels
			ax.set_ylabel('$\\xi$')
			ax.set_xlabel('$\\eta$')

			## grids

			# rho grid
			if grid == 'single' or grid == 'both' or grid == 'rho':
				ax.grid()
			
			# lon-lat grid
			if grid == 'both' or grid == 'lonlat':
				lon_levels = np.linspace(np.nanmin(lon), np.nanmax(lon), 10)
				lat_levels = np.linspace(np.nanmin(lat), np.nanmax(lat), 10)
				cax_eta = ax.contour(etas, xis, lon, colors='grey', linewidths=1)
				cax_xi = ax.contour(etas, xis, lat, colors='grey', linewidths=1)
				if second_grid_labels:
					ax.clabel(cax_eta, fmt="%3.2f")
					ax.clabel(cax_xi, fmt="%3.2f")

	# colorize land masses
	if land_gray:
		y = np.ones_like(grid_data.mask_rho.values)
		y[grid_data.mask_rho.values == 1] = np.nan
		plot_data(grid_data, y, lon_name='lon_rho', lat_name='lat_rho', ax=ax, vmin=0, vmax=2, cmap='gray', colorbar=False, as_contourfill=as_contourfill)

	ax.set_title(title)

	# add a colorbar
	if not is_animation_frame and colorbar:
		cbar = plt.colorbar(cax, ax=ax, label=cbar_label, ticks=cbar_ticks)

	# call plot.show
	if call_show:
		plt.show()

	# return plot data and figures themselve to use for colorbars
	return data_main if highlight_subdomain is None else data_subd[highlight_subdomain], plots



def plot_block(ax, data_block,
	# data selection
	surface_range=slice(0, 1),
	eta_range=slice(0, 1),
	xi_range=slice(-1, None),
	# contours on surface
	surface_contours=None, 
	surface_contours_kwargs={},
	# contours on side
	side_contours=None, 
	side_contours_kwargs={},
	# levels and extension
	r=1,
	z_level=None,
	# off for distance between plots
	off=0.05, zoff=None,
	# kwargs for contourf
	**kwargs
	):
	"""
	Plot 3D data by showing a cube where the surfaces correspond to the mean along the perpendicular axis.
	"""
	
	# create meshs
	xx, yx = np.meshgrid(np.linspace(-r, r, data_block.shape[1]), np.linspace(-r, r, data_block.shape[2]))
	z = np.linspace(0, 1, data_block.shape[0]) if z_level is None else -1*z_level
	xz, zz = np.meshgrid(np.linspace(-r, r, data_block.shape[1]), z)
	
	# get location of surface
	zoff = off if zoff is None else zoff
	surface_z = z.max() + zoff

	# calculate the different averages
	dsurface = np.nanmean(data_block[surface_range], axis=0)
	dsideeta = np.nanmean(data_block[:, eta_range], axis=1)
	dsidexi = np.nanmean(data_block[:, :, xi_range], axis=2)

	# to return
	cax1, cax2, cax3 = None, None, None

	# calculate vmin, vmax
	if 'vmin' not in kwargs:
		kwargs['vmin'] = min(np.min(dsurface), np.min(dsideeta), np.min(dsidexi))
	if 'vmax' not in kwargs:
		kwargs['vmax'] = max(np.max(dsurface), np.max(dsideeta), np.max(dsidexi))

	# plot top
	dcbar = np.array([[kwargs['vmin'], kwargs['vmin']], [kwargs['vmin'], kwargs['vmax']]])
	cbarx, cbary = np.meshgrid(np.linspace(-2*r, -1.5*r, 2), np.linspace(-2*r, -1.5*r, 2))
	cax1 = ax.contourf(cbarx, cbary, dcbar, zdir='z', offset=surface_z, **kwargs)
	
	# plot data
	ax.contourf(xx, yx, dsurface, zdir='z', offset=surface_z, zorder=1, **kwargs)
	ax.contourf(xz, dsideeta, zz, zdir='y', offset=-r - off, zorder=1, **kwargs)
	ax.contourf(dsidexi, xz, zz, zdir='x', offset=r + off, zorder=1, **kwargs)
	
	# plot contours
	if surface_contours is not None:
		# again, vmin vmax for sides
		if 'vmin' not in surface_contours_kwargs:
			surface_contours_kwargs['vmin'] = np.min(surface_contours)
		if 'vmax' not in surface_contours_kwargs:
			surface_contours_kwargs['vmax'] = np.max(surface_contours)
		# plot sides
		dcbar = np.array([[surface_contours_kwargs['vmin'], surface_contours_kwargs['vmin']], [surface_contours_kwargs['vmin'], surface_contours_kwargs['vmax']]])
		cax2 = ax.contour(cbarx, cbary, dcbar, zdir='z', offset=surface_z, **surface_contours_kwargs)
		ax.contour(xx, yx, surface_contours, zdir='z', offset=surface_z, **surface_contours_kwargs)
	# if contours on the side
	if side_contours is not None:
		cax3 = ax.contour(xz, side_contours[0], zz, zdir='y', offset=-r - off, **side_contours_kwargs)
		ax.contour(side_contours[1], xz, zz, zdir='x', offset=r + off, **side_contours_kwargs)
	
	# mark selected range with red lines
	u0 = z[surface_range]
	u1 = np.linspace(-r, r, data_block.shape[1])[eta_range]
	u2 = np.linspace(-r, r, data_block.shape[2])[xi_range]
	ax.plot([r+off for _ in u0], [r+off for _ in u0], u0, c='red', lw=5)
	ax.plot([-r-off for _ in u1], u1, [z[0] for _ in u1], c='red', lw=5)
	ax.plot(u2, [r+off for _ in u2], [z[0] for _ in u2], c='red', lw=5)

	# plot help lines
	line_res = 400
	xxsurf, yxsurf = np.meshgrid(np.linspace(-r, r, line_res), np.linspace(-r, r, line_res))
	xxside, yxside = np.meshgrid(np.linspace(-r, r, line_res), np.linspace(z.min(), z.max(), line_res))
	
	# plot center grid line
	surf_map_main = np.zeros((line_res, line_res))
	surf_map_main[line_res//2, :] = 1
	surf_map_main[:, line_res//2] = 1
	surf_side_main = np.zeros((line_res, line_res))
	surf_side_main[:, line_res//2] = 1
	ax.contourf(xxsurf, yxsurf, surf_map_main, zdir='z', offset=surface_z, zorder=1, levels=1, vmin=0, vmax=1, colors=[(0, 0, 0, 0), (0, 0, 0, 0.5)])
	ax.contourf(xxside, surf_side_main, yxside, zdir='y', offset=-r-off, zorder=1, levels=1, vmin=0, vmax=1, colors=[(0, 0, 0, 0), (0, 0, 0, 0.5)])
	ax.contourf(surf_side_main, xxside, yxside, zdir='x', offset=r+off, zorder=1, levels=1, vmin=0, vmax=1, colors=[(0, 0, 0, 0), (0, 0, 0, 0.5)])
	
	# plot outer grid lines
	if r >= 2:
		surf_map_main = np.zeros((line_res, line_res))
		surf_map_main[line_res//4, :] = 1
		surf_map_main[:, line_res//4] = 1
		surf_map_main[3*line_res//4, :] = 1
		surf_map_main[:, 3*line_res//4] = 1
		surf_side_main = np.zeros((line_res, line_res))
		surf_side_main[:, line_res//4] = 1
		surf_side_main[:, 3*line_res//4] = 1
		ax.contourf(xxsurf, yxsurf, surf_map_main, zdir='z', offset=surface_z, zorder=1, levels=1, vmin=0, vmax=1, colors=[(0, 0, 0, 0), (0, 0, 0, 0.2)])
		ax.contourf(xxside, surf_side_main, yxside, zdir='y', offset=-r-off, zorder=1, levels=1, vmin=0, vmax=1, colors=[(0, 0, 0, 0), (0, 0, 0, 0.2)])
		ax.contourf(surf_side_main, xxside, yxside, zdir='x', offset=r+off, zorder=1, levels=1, vmin=0, vmax=1, colors=[(0, 0, 0, 0), (0, 0, 0, 0.2)])

	# set limits
	ax.set_xlim(-r, r)
	ax.set_ylim(-r, r)
	ax.set_zlim(z.min(), surface_z)

	# set background color to be invisible
	ax.xaxis.pane.fill = False
	ax.yaxis.pane.fill = False
	ax.zaxis.pane.fill = False
	ax.xaxis.pane.set_edgecolor('w')
	ax.yaxis.pane.set_edgecolor('w')
	ax.zaxis.pane.set_edgecolor('w')

	# define where to show axis and ticks
	ax.xaxis._axinfo['juggled'] = (2,0,1)
	ax.yaxis._axinfo['juggled'] = (2,1,0)

	# labels
	ax.set_xlabel('$\\xi$')
	ax.set_ylabel('$\\eta$')
	ax.set_zlabel('depth')

	# show only radii as ticks
	ax.set_xticks([i for i in range(-r, r+1)])
	ax.set_yticks([i for i in range(-r, r+1)])

	return cax1, cax2, cax3