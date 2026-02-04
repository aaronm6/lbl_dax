"""
Utilities for analysis, like plotting routines
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle

__all__ = ['plot_top_pattern']

dim_2x2 = 7.85 # mm
cn_2x2 = np.array([
   [1,1],
   [1,-1],
   [-1,-1],
   [-1,1]]) * dim_2x2

cn_ch_mm = np.array([
    [.25,.25],
    [.25,-6.15-.25],
    [-6.15-.25, -6.15-.25],
    [-6.15-.25,.25]])
ch_corner_pos_mm = np.empty((16,2))
for k4 in range(4):
    for k1 in range(4):
        ch_corner_pos_mm[4*k4+k1,:] = cn_2x2[k4,:] + cn_ch_mm[k1,:]
a_ch = 6.15 # mm

ch_patches = [Rectangle(tuple(ch_corner_pos_mm[k,:]), a_ch, a_ch) for k in range(16)]

def plot_top_pattern(hit_i, ax=None, cmap=None, add_circ=True):
    """
    plot_top_pattern
    
    Take an array of 16 hit values (one for each SiPM in an array), plot the hit pattern
    with boxes shaded with a color scale.  Input `hit_i` should be a 1d, length-16 array.
    
    Optionally, this script can plot a circle the same size as the diameter of the TPC
    i.e. 30mm, with keyword argument 'add_circ'. (default is True).
    """
    if ax is None:
        ax = plt.gca()
    hit_collection = PatchCollection(ch_patches, array=np.asarray(hit_i).squeeze(), cmap=cmap)
    ax.add_collection(hit_collection)
    plt.sci(hit_collection)
    ax.autoscale_view()
    if add_circ:
        tpc_circle = Circle((0., 0.), radius=15., facecolor='none', edgecolor='r',lw=.5)
        ax.add_patch(tpc_circle)
    return hit_collection





