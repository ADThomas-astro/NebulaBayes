from __future__ import print_function, division
from collections import OrderedDict as OD
import numpy as np
import pandas as pd
# For interpolating an n-dimensional regular grid:
from scipy.interpolate import RegularGridInterpolator as siRGI
import itertools # For Cartesian product


"""
This module contains functions to load the model grid database table, constuct
model flux arrays, and interpolate those arrays to higher resolution.

ADT 2015 - 2017

"""



class Grid_description(object):
    """
    Class to hold information about N-dimensional arrays - the names of the
    parameters corresponding to each dimension, the values of the parameters
    along each dimension, etc.
    """
    def __init__(self, param_names, param_value_arrs):
        """
        Initialise an instance with useful attributes, including mappings
        between important quantities that define the grid.
        param_names: List of parameter names as strings
        param_value_arrs: List of lists of parameter values over the grid,
                          where sublists correspond to param_names.
        Note that NebulaBayes code relies on the dictionaries below being ordered.
        """
        assert len(param_names) == len(param_value_arrs)
        # Record some basic info
        self.param_names = param_names
        self.param_values_arrs = param_value_arrs
        self.ndim = len(param_names)
        self.shape = tuple( [len(arr) for arr in param_value_arrs ] )
        self.n_gridpoints = np.product( self.shape )

        # Define mappings for easily extracting data about the grid
        self.paramName2ind = OD(zip(param_names, range(self.ndim)))
        #self.ind2paramName = OD(zip(range(self.ndim), param_names))
        self.paramName2paramValueArr = OD(zip(param_names, param_value_arrs))
        # self.ind2paramValueArr = OD(zip(range(self.ndim), param_value_arrs))

        self.paramNameAndValue2arrayInd = OD()
        for p, arr in self.paramName2paramValueArr.items():
            for i,v in enumerate(arr):
                self.paramNameAndValue2arrayInd[(p,v)] = i
        # So self.paramNameAndValue2ArrayInd[(p,v)] will give the index along
        # the "p" axis where the parameter with name "p" has value v.

        self.paramName2paramMinMax = OD( (p,(a.min(), a.max())) for p,a in 
                                          self.paramName2paramValueArr.items() )



class NB_Grid(Grid_description):
    """
    Simple class to hold n_dimensional grid arrays, along with a description of
    the grid.
    Will hold an grid of the same shape for each emission line.
    """
    def __init__(self, param_names, param_value_arrs):
        """ Initialise """
        super(NB_Grid, self).__init__(param_names, param_value_arrs)
        self.grids = OD()



def initialise_grids(grid_file, grid_params, lines_list, interpd_grid_shape):
    """
    Initialise grids and return a simple object, which will have
    attributes Params, Raw_grids and Interpd_grids.
    The returned object 
    contains all necessary grid information for NebulaBayes, and may be used as an 
    input to repeated NebulaBayes runs to avoid recalculation of grid data in each run.
    The Raw_grids and Interpd_grids attributes are instances of the NB_Grid
    class defined in this module.  The Params attribute is an instance of the
    Grid_parameters class defined in this module.
    grid_file:  the filename of an ASCII csv table containing photoionisation model
                grid data in the form of a database table.
                Each gridpoint (point in parameter space) is a row in this table.
                The values of the grid parameters for each row are defined in a column
                for each parameter.
                No assumptions are made about the order of the gridpoints (rows) in the table.
                Spacing of grid values along an axis may be uneven, 
                but the full grid is required to a be a regular, n-dimensional rectangular grid.
                There is a column of fluxes for each modelled emission line, and model fluxes
                are assumed to be normalised to Hbeta
                Any non-finite fluxes (e.g. nans) will be set to zero.
    grid_params: List of the unique names of the grid parameters as strings.
                 The order is the order of the grid dimensions, i.e. the order
                 in which arrays in NebulaBayes will be indexed.
    interpd_grid_shape: A tuple of integers, giving the size of each dimension
                        of the interpolated grid.  The order of the integers
                        corresponds to the order of parameters in grid_params.
                        The default is 30 gridpoints along each dimension.  Note
                        that the number of interpolated gridpoints entered in
                        interpd_grid_shape may have a major impact on the speed
                        of the program.
    """
    print("Loading input grid table...")
    # Load database csv table containing the model grid output
    DF_grid = pd.read_table(grid_file, header=0, delimiter=",")
    print("Cleaning input grid table...")
    # Remove any whitespace from column names
    DF_grid.rename(inplace=True, columns={c:c.strip() for c in DF_grid.columns})
    for line in lines_list: # Ensure line columns have float dtype
        DF_grid[line] = pd.to_numeric(DF_grid[line], errors="coerce")
        # We "coerce" errors to NaNs, which will be set to zero

    # Clean and check the model data:
    for line in lines_list:
        # Check that all emission lines in input are also in the model data:
        if not line in DF_grid.columns:
            raise ValueError("Measured emission line " + line +
                             " was not found in the model data.")
        # Set any non-finite model fluxes to zero:
        ####### Dodgy?!?!
        DF_grid.loc[~np.isfinite(DF_grid[line].values), line] = 0
        # pandas complains about DF_grid[line][ ~np.isfinite( DF_grid[line].values ) ] = 0
        # Check that all model flux values are non-negative:
        if np.sum( DF_grid[line].values < 0 ) != 0:
            raise ValueError("A flux value for modelled emission line " +
                             line + " is negative.")

    #--------------------------------------------------------------------------
    # Construct raw flux grids
    Raw_grids = construct_raw_grids(DF_grid, grid_params, lines_list)

    #--------------------------------------------------------------------------
    # Interpolate flux grids
    Interpd_grids = interpolate_flux_arrays(Raw_grids, interpd_grid_shape)

    return Raw_grids, Interpd_grids



def construct_raw_grids(DF_grid, grid_params, lines_list):
    """
    Construct arrays of flux grids from the input flux table.
    DF_grid: pandas DataFrame table holding the predicted fluxes of the model grid.
    Params:  Object holding parameter names, corresponding to columns in DF_grid.
    lines_list: list of names of emission lines of interest, corresponding to
                columns in DF_grid.
    """
    # Set up raw grid...

    # Determine the list of parameter values for the raw grid:
    # List of arrays; each array holds the grid values for a parameter:
    param_val_arrs_raw = []
    for p in grid_params:
        # Ensure we have a sorted list of unique values for each parameter:
        param_val_arrs_raw.append( np.sort( np.unique( DF_grid[p].values ) ) )
    # Initialise a grid object to hold the raw grids:
    Raw_grids = NB_Grid(grid_params, param_val_arrs_raw)

    # Check that the input database table is the right length:
    # (This is equivalent to checking that we have a rectangular grid, e.g.
    # without missing values.  The spacing does not need to be uniform.)
    if Raw_grids.n_gridpoints != len(DF_grid):
        raise ValueError("The input model grid table does not" + 
                         "have a consistent length.")

    #--------------------------------------------------------------------------
    # Construct the raw model grids as a multidimensional array for each line
    print("Building flux arrays for the model grids...")
    # We use an inefficient method for building the model grids because we're
    # not assuming anything about the order of the rows in the input table.
    # First reduce DF_grid to include only the required columns:
    columns = grid_params + lines_list
    DF_grid = DF_grid[ columns ]
    for emission_line in lines_list: # Initialise new (emission_line,flux_array)
        # item in dictionary, as an array of nans:
        Raw_grids.grids[emission_line] = np.zeros( Raw_grids.shape ) + np.nan
    # Iterate over rows in the input model grid table:
    for row_tuple in DF_grid.itertuples(index=False, name=None):
        # row_tuple is not a namedtuple, since I set name=None.  I don't want
        # namedtuples => columns names would need to be valid python identifiers
        # and there would be a limit of 255 columns
        row_vals = dict(zip(columns,row_tuple)) # Maps col names to row values
        # Generate the value of each grid parameter for this row (in order)
        row_p_vals = ( (p, row_vals[p]) for p in grid_params )
        # List the grid indices associated with the param values for this row
        row_p_inds = [Raw_grids.paramNameAndValue2arrayInd[(p,v)] for p,v in row_p_vals]
        for line in lines_list: # Iterate emission lines
            # Write the line flux value for this gridpoint into the correct
            # location in the flux array for this line:
            Raw_grids.grids[line][tuple(row_p_inds)] = row_vals[line]

    arr_n_bytes = Raw_grids.grids[lines_list[0]].nbytes
    n_lines = len(lines_list)
    print( """Number of bytes in raw grid flux arrays: {0} for 1 emission line, 
    {1} total for all {2} lines""".format( arr_n_bytes, arr_n_bytes*n_lines,
                                                                    n_lines ) )

    return Raw_grids



def interpolate_flux_arrays(Raw_grids, interpd_shape):
    """
    Interpolate emission line grids.  Negative values arising from
    interpolation are set to zero (but use of the Akima splines should prevent
    bad "overshoots" in interpolation).

    """
    print("Interpolating model emission line flux grids to shape {0}...".format(
                                                          tuple(interpd_shape)))

    # Initialise NB_Grid object for interpolated arrays
    val_arrs_interp = []
    for p, n in zip(Raw_grids.param_names, interpd_shape):
        p_min, p_max = Raw_grids.paramName2paramMinMax[p]
        val_arrs_interp.append( np.linspace(p_min, p_max, n) )
    Interpd_grids = NB_Grid(list(Raw_grids.param_names), val_arrs_interp)

    # A list of all parameter value combinations in the
    # interpolated grid in the form of a numpy array:
    param_combos = np.array( list( itertools.product( *val_arrs_interp ) ) )
    # # A list of all index combinations for the
    # # interpolated grid, corresponding to param_combos:
    # param_index_combos = np.array( list(
    #         itertools.product( *[np.arange(n) for n in Interpd_grids.shape] ) ) )
    # # Generate a tuple containing an index array for each grid dimension,
    # # with each combination of indices (e.g. from the 7th entry in each
    # # index array) correspond to a param_combos row (e.g. the 7th):
    # param_fancy_index = tuple(
    #         [ param_index_combos[:,i] for i in np.arange(Params.n_params) ] )
    # # This will be used to take advantage of numpy's fancy indexing capability.

    for emission_line, raw_flux_arr in Raw_grids.grids.items():
        print("Interpolating for {0}...".format(emission_line))
        # Interpd_grids.grids[emission_line], val_arrs_interp = interpolate_Akima_nD(raw_flux_arr,
        #                                                 val_arrs, interpd_shape)

        # Create new (emission_line, flux_array) item in dictionary of interpolated grid arrays:
        # Interpd_grids.grids[emission_line] = np.zeros( interpd_shape )
        # Create function for carrying out the interpolation:
        interp_fn = siRGI(tuple(Raw_grids.param_values_arrs),
                          Raw_grids.grids[emission_line], method="linear")
        # Fill the interpolated fluxes into the final grid structure, using "fancy indexing":
        # Interpd_grids.grids[emission_line][param_fancy_index] = interp_fn( param_combos )
        Interpd_grids.grids[emission_line] = interp_fn(param_combos).reshape(interpd_shape)
        # I'm gambling here that "reshape" works the way I want... seems like it does!
        # But the fancy indexing seems to be faster...

    n_lines = len(Interpd_grids.grids)
    line_0, arr_0 = list(Interpd_grids.grids.items())[0]
    print( """Number of bytes in interpolated grid flux arrays: {0} for 1 emission line, 
    {1} total for all {2} lines""".format( arr_0.nbytes, arr_0.nbytes*n_lines,
                                                                      n_lines ) )

    # Set negative values to zero:
    for a in Interpd_grids.grids.values():
        a[a < 0] = 0

    return Interpd_grids


