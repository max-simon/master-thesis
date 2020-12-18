#!/usr/bin/env python
from netCDF4 import Dataset  # pylint: disable=no-name-in-module
import numpy as np

#########################################################
# Class for ROMS grd and clm files
# (For use in various post-processing scripts)
#########################################################

class getGrid(object):
	'''
	Read the basics of ROMS setup into class for further use in other functions
	and classes.
	'''
	# Read grid file
	def __init__(self,grdfile):
		# Set grd file
		self.grdfile = grdfile
		self.ncgrd = Dataset(grdfile, mode='r')
		# Read mask
		self.mask_rho = self.ncgrd.variables['mask_rho'][:]
		self.FillValue = getattr(self.ncgrd.variables['mask_rho'],'_FillValue')
	# Read dimensions
		self.SY = self.mask_rho.shape[0]
		self.SX = self.mask_rho.shape[1]
		
	def getAttrs(self,clmfile):
		# Set clm file
		self.ncclm  = Dataset(clmfile, mode='r')
		# Read attributes
		try:
			self.theta_s = getattr(self.ncclm,'theta_s')
			self.theta_b = getattr(self.ncclm,'theta_b')
			self.hc      = getattr(self.ncclm,'hc')
		except AttributeError:
			self.theta_s = self.ncclm.variables['theta_s'][0]
			self.theta_b = self.ncclm.variables['theta_b'][0]
			self.hc      = self.ncclm.variables['hc'][0]            
		# Vertical dimension
		self.NZ       = self.ncclm.dimensions['s_rho'].size
	
	def setClmFiles(self,clmfile,clm2file):
	# Set clm file
		if not hasattr(self, 'ncclm'):
			self.ncclm  = Dataset(clmfile, mode='r')
		# Set clm2 file
		self.ncclm2 = Dataset(clm2file, mode='r')

	def getTopo(self):
		# Read topography
		self.h     = self.ncgrd.variables['h'][:]
		self.hmin  = getattr(self.ncgrd,'hmin')
		self.hmax  = getattr(self.ncgrd,'hmax')
		
	def getLatLon(self):
		# Read Lat/Lon
		self.lon_rho  = self.ncgrd.variables['lon_rho'][:]
		self.lat_rho  = self.ncgrd.variables['lat_rho'][:]
		
	def getArea(self):
		# Read pm/pn
		self.area  = 1/(self.ncgrd.variables['pm'][:]*self.ncgrd.variables['pn'][:])

	def getAngle(self):
		# Read angle
		self.angle  = self.ncgrd.variables['angle'][:]

#########################################################
# Vertical sigma level depths and spacing
#########################################################

def compute_zlev(fpin,fpin_grd,NZ,type,zeta=None,stype=3):
	# Compute z levels of rho points for ZERO SSH. Input:
	#
	#  fpin: file descriptor pointing to a NetCDF file containing theta_b,
	#        theta_s and Tcline or hc
	#  fpin_grd: file descriptor pointing to a NetCDF file containing h
	#  NZ: number of vertical (rho) levels
	#  type:  'r': rho points
	#         'w': w points
	#  stype: specifies type of sigma levels used:
	#          1: similar to Song, Haidvogel 1994
	#          2: Shchepetkin 2006
	#          3: Shchepetkin 2010 (or so)

	import numpy as np
	import sys
	
	h = fpin_grd.variables['h'][:,:]
	try:
		theta_b = fpin.theta_b
		theta_s = fpin.theta_s
	except AttributeError:
		# theta_b/s may be variables:
		theta_b = fpin.variables['theta_b'][0]
		theta_s = fpin.variables['theta_s'][0]
		
	if stype == 1:
		hmin = min(min(h))
		try:
			Tcline = fpin.Tcline
			hc = min(hmin,Tcline)
		except AttributeError:
			hc = fpin.hc
			hc = min(hmin,hc)
	elif stype == 2 or stype == 3:
		try:
			hc = fpin.hc
		except AttributeError:
			# hc may be a variable:
			hc = fpin.variables['hc'][0]
	else:
		msg = '{}: Unknown type of sigma levels'.format(stype)
		sys.exit(msg)
	ds = 1./NZ  # float, to prevent integer division in sc
	if type == 'w':
		lev = np.arange(NZ+1)
		sc = (lev - NZ) * ds
		nr_zlev = NZ+1 # number of vertical levels
	else:
		lev = np.arange(1,NZ+1)
		sc = -1 + (lev-0.5)*ds
		nr_zlev = NZ # number of vertical levels
	Ptheta = np.sinh(theta_s*sc)/np.sinh(theta_s)
	Rtheta = np.tanh(theta_s*(sc+.5))/(2*np.tanh(.5*theta_s))-.5
	if stype <= 2:
		Cs = (1-theta_b)*Ptheta+theta_b*Rtheta
	elif stype == 3:
		if theta_s > 0:
			csrf=(1.-np.cosh(theta_s*sc))/(np.cosh(theta_s)-1.)
		else:
			csrf=-sc**2
		if theta_b > 0:
			Cs=(np.exp(theta_b*csrf)-1.)/(1.-np.exp(-theta_b))
		else:
			Cs=csrf
	z0 = np.zeros((nr_zlev,h.shape[0],h.shape[1]),np.float)
	if stype == 1:
		cff = (sc-Cs)*hc
		cff1 = Cs
		hinv = 1.0 / h
		for k in range(nr_zlev):
			z0[k,:,:] = cff[k]+cff1[k]*h
			if not (zeta is None):
				z0[k,:,:] = z0[k,:,:]+zeta*(1.+z0[k,:,:]*hinv)
	elif stype == 2 or stype == 3:
		hinv = 1.0/(h+hc)
		cff = hc*sc
		cff1 = Cs
		for k in range(nr_zlev):
			tmp1 = cff[k]+cff1[k]*h
			tmp2 = np.multiply(tmp1,hinv)
			if zeta is None:
				z0[k,:,:] = np.multiply(h,tmp2)
			else:
				z0[k,:,:] = zeta + np.multiply((zeta+h),tmp2)
	# Return
	return z0

def compute_dz(fpin,fpin_grd,NZ,zeta=None,stype=3):
  
	# Compute dz of sigma level rho points for ZERO SSH. Input:
	#
	#  fpin: file descriptor pointing to a NetCDF file containing theta_b,
	#        theta_s and Tcline or hc
	#  fpin_grd: file descriptor pointing to a NetCDF file containing h
	#  NZ: number of vertical (rho) levels
	#  stype: specifies type of sigma levels used:
	#          1: similar to Song, Haidvogel 1994
	#          2: Shchepetkin 2006
	#          3: Shchepetkin 2010 (or so)
	
	# Compute depth of w sigma levels
	depth_w = -compute_zlev(fpin,fpin_grd,NZ,type='w',zeta=zeta,stype=3)
	
	# Compute dz between w sigma levels (= dz of sigma layer)
	dz_sigma = depth_w[:-1]-depth_w[1:]
	
	return dz_sigma



#########################################################
# Additions from Max Simon
# Author: Max Simon
# Year: 2020
#########################################################

def get_cell_heights(z_values, depth):
	"""
	Structure if depth is False:

	-------------  // surface, top second cell
		  x        // rho point, idx 2
	-------------  // top first cell, bottom second cell

		  x        // rho point, idx 1

	-------------  // top zero-th cell, bottom first cell


		  x        // rho point, idx 0


	-------------  // ground, bottom zero-th cell

	 Structure if depth is True

	-------------  // surface, top zero-th cell
		  x        // depth point, idx 0
	-------------  // top first cell, bottom zero-th cell

		  x        // depth point, idx 1

	-------------  // top second cell, bottom first cell


		  x        // depth point, idx 2


	-------------  // ground, bottom second cell

	Idea:
		- loop from top to bottom (this means for depth = False from last index to first)
		- calculate distance from current point to last_depth  --> half the cell height
		- last_depth is initially 0 and set to _current rho point + half the cell height_ after each iteration
		- cell size is _2 x half the cell height_

	Note: if depth = False this has to be done for each grid point seperately!

	"""
	heights = np.zeros_like(z_values)
	last_height = 0.0 if depth else np.zeros((z_values.shape[1], z_values.shape[2]))
	zero_edge_case = False
	for srho_idx in range(z_values.shape[0]):
		# go from top to bottom
		srho = srho_idx if depth else (z_values.shape[0] - srho_idx - 1)
		# handle edge case:
		if srho == 0 and (z_values[srho] == 0).any():
			assert (z_values[srho] == 0).all()
			print('Zero Edge Case detected')
			zero_edge_case = True
			continue

		# calc dist to last height
		half = np.abs(z_values[srho]) - last_height
		# handle edge case
		if srho == 1 and zero_edge_case:
			half = 0.5*half
			previous_srho = 0 if depth else -1
			heights[previous_srho] = half
			zero_edge_case = False
			print('Zero Edge Case solved')

		assert np.array(half >= 0).all(), (srho_idx, srho, z_values[srho], last_height, half)
		heights[srho] = 2*half
		# update last_height
		last_height = np.abs(z_values[srho]) + half
	return heights


def create_zlevel_file(grid_path, sample_data_path, out_path):
	"""
	Create a netCDF file containing the zlevels
	"""
	sample_data = Dataset(sample_data_path)
	is_zslice_file = 'depth' in sample_data.dimensions

	if is_zslice_file:
		print('Sample Data is z sliced')
		z_levels = np.array(sample_data['depth'])
		z_thickness = get_cell_heights(z_levels, True)

		assert np.sum(z_thickness[:-1]) + 0.5*z_thickness[-1] == abs(z_levels[-1]), (np.sum(z_thickness[:-1]), z_thickness[-1], z_levels[-1]) 

		with Dataset(out_path, mode='w') as new_dataset:
			# copy global attributes all at once via dictionary
			new_dataset.createDimension('depth', len(z_levels))
			# save zlevels
			new_dataset.createVariable('z_level', np.float32, dimensions=('depth',))
			new_dataset['z_level'][:] = np.abs(z_levels)
			new_dataset.createVariable('thickness_z', np.float32, dimensions=('depth'))
			new_dataset['thickness_z'][:] = np.abs(z_thickness)

	else:
		sample_data.close()  # just make sure that we dont interfer with other routines
		print('Sample Data is raw ROMS output')
		# calculate the zlevels
		grid = Dataset(grid_path)
		sample_data = Dataset(sample_data_path)
		n_s_rho = sample_data.dimensions['s_rho'].size
		n_eta_rho = sample_data.dimensions['eta_rho'].size
		n_xi_rho = sample_data.dimensions['xi_rho'].size

		z_levels_rho = compute_zlev(sample_data, grid, n_s_rho, 'r')
		z_levels_w = compute_zlev(sample_data, grid, n_s_rho, 'w')
		z_thickness_rho = get_cell_heights(z_levels_rho, False)

		control = np.sum(z_thickness_rho, axis=0) - np.array(grid['h'])
		assert np.max(np.abs(control)) < 5, 'Height calculation differs more than 5m'

		with Dataset(out_path, mode='w') as new_dataset:
			# copy global attributes all at once via dictionary
			new_dataset.createDimension('s_rho', n_s_rho)
			new_dataset.createDimension('eta_rho', n_eta_rho)
			new_dataset.createDimension('xi_rho', n_xi_rho)
			new_dataset.createDimension('s_w', n_s_rho + 1)
			
			# save zlevels
			new_dataset.createVariable('z_level', np.float32, dimensions=('s_rho', 'eta_rho', 'xi_rho'))
			new_dataset['z_level'][:] = np.abs(z_levels_rho)
			new_dataset.createVariable('z_level_w', np.float32, dimensions=('s_w', 'eta_rho', 'xi_rho'))
			new_dataset['z_level_w'][:] = np.abs(z_levels_w)
			new_dataset.createVariable('thickness_z', np.float32, dimensions=('s_rho', 'eta_rho', 'xi_rho'))
			new_dataset['thickness_z'][:] = np.abs(z_thickness_rho)


if __name__ == "__main__":

	import argparse
	
	# create parser
	parser = argparse.ArgumentParser()
	# add arguments
	parser.add_argument('--input', type=str, required=True, help="Sample Input Path")
	parser.add_argument('--grid', type=str, required=True, help="Grid path")
	parser.add_argument('--output', type=str, help="Output path")
	args = parser.parse_args()

	# execute
	create_zlevel_file(args.grid, args.input, args.output)

