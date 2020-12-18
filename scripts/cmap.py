#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Max Simon
# Year: 2020


from matplotlib import colors
from matplotlib.cm import get_cmap
import numpy as np

# colormap like (Gruber, 2011)
COLORS_DIFF = [
    '#F8E14C',
    '#FFA63D',
    '#FF6637',
    '#FF7D83',
    '#FFFFFF',
    '#4262B3',
    '#2F62B4',
    '#3F96D2',
    '#5BD2E2'
]

# colormap like (Gruber, 2011)
COLORS_W2G = [
    '#FFFFFF',
    '#7D90CA',
    '#274BA7',
    '#512795',
    '#FF351A',
    '#FF712A',
    '#F8E754',
    '#79C651',
    '#2EAC61'
]

# colormap like (Gruber, 2011)
COLORS_G2R = [
    '#1F4F25',
    '#359842',
    '#5DBF49',
    '#B6D550',
    '#FF8027',
    '#E5401C',
    '#5E1010'
]

# create colormaps for matplotlib
DIFF = colors.LinearSegmentedColormap.from_list('diff', COLORS_DIFF)
DIFF_r = colors.LinearSegmentedColormap.from_list('diff', COLORS_DIFF[::-1])
DIFF.set_bad('gray')
DIFF_r.set_bad('gray')

W2G = colors.LinearSegmentedColormap.from_list('w2g', COLORS_W2G)
W2G_r = colors.LinearSegmentedColormap.from_list('w2g_r', COLORS_W2G[::-1])
W2G.set_bad('gray')
W2G_r.set_bad('gray')

G2R = colors.LinearSegmentedColormap.from_list('g2r', COLORS_G2R)
G2R_r = colors.LinearSegmentedColormap.from_list('g2r_r', COLORS_G2R[::-1])
G2R.set_bad('gray')
G2R_r.set_bad('gray')

def get_step_cmap(cmap, num_values):
    # create a colormap with steps instead of continous colors
    mycmap = get_cmap(cmap, num_values)
    new_colors = mycmap(np.linspace(0, 1, num_values))
    return colors.ListedColormap(new_colors, 'stepwise')
