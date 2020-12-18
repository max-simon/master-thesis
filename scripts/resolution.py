#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


import sys
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')
import numpy as np
import nominal_resolution as nr

import xarray as xr
from romstools.utils import get_area_map


if __name__ == "__main__":

	run = sys.argv[1]
	print(run)

	# load grid
	grid = xr.open_dataset('/nfs/kryo/work/maxsimon/data/'+run+'/grid.nc')
	# load grid_data to get mask for subdomain
	grid_data = np.load('/nfs/kryo/work/maxsimon/data/'+run+'/grid.npz')
	gruber_mask = grid_data['gruber_mask']

	# calculate cell areas
	area = get_area_map(grid)
	# get lon and lat
	lon = grid.lon_psi.values
	lat = grid.lat_psi.values

	areas = []
	lats = []
	lons = []

	# convert data to format accepted by nominal_resolution, this is a flat list of all cells with their four points around
	for x in range(1, area.shape[0] - 1):
		for y in range(1, area.shape[1] - 1):
			# skip out of mask
			if not gruber_mask[x, y]:
				continue

			areas.append(area[x, y])
			lats.append([
				lat[x-1, y-1],
				lat[x, y-1],
				lat[x, y],
				lat[x-1, y]
			])
			lons.append([
				lon[x-1, y-1],
				lon[x, y-1],
				lon[x, y],
				lon[x-1, y]
			])

	# convert to numpy arrays
	areas = np.array(areas)
	lats = np.array(lats)
	lons = np.array(lons)

	# calculate nominal resolutions
	# NOTE: this is different from the reported mean resolution as this always take the largest distance in a grid cell (so diagonals)
	mean_res = nr.mean_resolution(areas, lats, lons, True)
	print(mean_res)
	nom_res = nr.nominal_resolution(mean_res)
	print(nom_res)
