import sys
sys.path.append('/nfs/kryo/work/maxsimon/master-thesis/scripts')

from romstools.utils import parse_slice


def check_slice_single(s):
	"""
	Raise an error if a slice would lead to a single value.
	Note that this can happen anyway as it also depends on the size of the array!
	"""
	if (s.start is None and s.stop == 1) or (s.start is not None and s.stop == s.start + 1):
		raise RuntimeError("The specified slice returns only a single item. Because there grids inside, this is not allowed.")


def get_grid_slices(eta_rho_slice, xi_rho_slice, s_rho_slice, **kwargs):
	"""
	Get the correct slices for v-grid, u-grid, psi-grid and s_w grid.
	kwargs can be used to specify which dimensions should be included in the output, this is useful for xarrays sel method.
	"""
	if eta_rho_slice is None:
		eta_rho_slice = slice(None, None)
	if xi_rho_slice is None:
		xi_rho_slice = slice(None, None)

	check_slice_single(eta_rho_slice)
	check_slice_single(xi_rho_slice)
	# we dont need to check s_rho because s_w will be larger and s_rho = 1 and depth = 1 is fine

	# set up slices
	slices = {
		'eta_rho': eta_rho_slice,
		'xi_rho': xi_rho_slice
	}
	# v-grid
	slices['eta_v'] = slice(eta_rho_slice.start, None) if eta_rho_slice.stop is None else slice(eta_rho_slice.start, eta_rho_slice.stop - 1)
	slices['xi_v'] = xi_rho_slice
	# u-grid
	slices['eta_u'] = eta_rho_slice
	slices['xi_u'] = slice(xi_rho_slice.start, None) if xi_rho_slice.stop is None else slice(xi_rho_slice.start, xi_rho_slice.stop - 1)
	# psi-grid
	slices['eta_psi'] = slices['eta_v']
	slices['xi_psi'] = slices['xi_u']

	# slice depth
	if s_rho_slice is not None:
		# we set both as it is a common use case
		# one of them should be filtered out by kwargs
		slices['s_rho'] = s_rho_slice
		slices['depth'] = s_rho_slice
		# s_w is larger than s_rho
		slices['s_w'] = slice(s_rho_slice.start, None) if s_rho_slice.stop is None else slice(s_rho_slice.start, s_rho_slice.stop + 1)

	# return data
	if len(kwargs) == 0:
		return slices
	else:
		selection = {key: slices[key] for key in slices if key in kwargs and kwargs[key]}
		return selection


def slice_on_rho_grid(xarr, eta_rho_slice, xi_rho_slice, s_rho_slice=None):
	"""
	Slice an xarray dataset on rho grid.
	"""
	dims = {key: True for key in xarr.dims.keys()}
	slices = get_grid_slices(eta_rho_slice, xi_rho_slice, s_rho_slice, **dims)
	return xarr.isel(**slices)



if __name__ == "__main__":

	import xarray as xr
	import argparse

	# create parser
	parser = argparse.ArgumentParser()

	# slices
	parser.add_argument('--eta-rho', type=parse_slice, help="Eta Rho Slice", default=slice(None, None))
	parser.add_argument('--xi-rho', type=parse_slice, help="Xi Rho Slice", default=slice(None, None))
	parser.add_argument('--s-rho', type=parse_slice, help="S Rho Slice", default=slice(None, None))
	# io
	parser.add_argument('--input', type=str, required=True, help="Input path")
	parser.add_argument('--output', type=str, help="Output path")

	args = parser.parse_args()

	# load dataset and extract dimensions
	data = xr.open_dataset(args.input)
	dims = {key: True for key in data.dims.keys()}

	# get the correct slices
	res = get_grid_slices(args.eta_rho, args.xi_rho, args.s_rho, **dims)

	# build the commmand
	cmd = 'ncks'

	for dim in res:
		# if full, dont put it to the command
		if res[dim].start is None and res[dim].stop is None:
			continue
		# start from beginning if None
		start = 0 if res[dim].start is None else res[dim].start
		# write nothing to set to the end else substract 1 because ncks is inclusive (in contrast to python)
		end = '' if res[dim].stop is None else res[dim].stop - 1

		if start == end:  # single value
			cmd += ' -d {:s},{:d}'.format(dim, start)
		else:  # slice
			cmd += ' -d {:s},{:d},{}'.format(dim, start, end)

	# put io
	cmd += ' '+args.input
	if args.output is not None:
		cmd += ' '+args.output

	print(cmd)