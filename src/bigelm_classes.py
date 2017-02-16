from __future__ import print_function, division
from collections import OrderedDict as OD
import numpy as np



"""
Adam D. Thomas 2015 - 2016



"""


#============================================================================
class Bigelm_grid(object):
    """
        Simple class to hold n_dimensional grid arrays,
        along with corresponding lists of values of the grid parameters.
        Will hold an grid of the same shape for each emission line.
    """
    def __init__(self, val_arrs):
        """
        Initialise the list of arrays listing the grid
        parameters, as well as some other useful quantities
        including the dictionary to hold the actual grids.
        """
        self.val_arrs = val_arrs # List of arrs of gridpoint vals for each param
        self.p_minmax = [(a.min(), a.max()) for a in val_arrs]
        self.par_indices = []
        for a in self.val_arrs:
            self.par_indices.append( {val:j for j,val in enumerate(a)} )
        # So self.par_indices[j][v] will give the index along axis j of a grid
        # that corresponds to parameter j having value v.
        # Calculate shape of each grid (length of each dimension):
        self.shape = tuple( [len(val_arr) for val_arr in self.val_arrs ] )
        self.ndim = len( self.shape )
        # Calculate the number of gridpoints (assuming a rectangular grid):
        self.n_gridpoints = np.product( self.shape )
        # Initialise a dictionary to hold the n-dimensional
        # model grid for each emission line:
        self.grids = OD()



