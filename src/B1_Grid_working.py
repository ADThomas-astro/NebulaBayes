from __future__ import print_function, division
from collections import OrderedDict as OD
# import sys
import numpy as np
import pandas as pd
# from scipy import ndimage
# For interpolating an n-dimensional regular grid:
from scipy.interpolate import RegularGridInterpolator as siRGI
# from scipy.interpolate import interp1d
# from scipy.interpolate import Akima1DInterpolator#, RectBivariateSpline
import itertools # For Cartesian product


"""
This module contains functions to load the model grid database table, constuct
model flux arrays, and interpolate those arrays to higher resolution.

ADT 2015 - 2017

"""



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


class dummy(object):
    pass


def initialise_grids(grid_file, grid_params, lines_list, interpd_grid_shape):
    """
    Initialise grids and return a simple object, which will have
    attributes Params, Raw_grids and Interpd_grids.
    The returned object 
    contains all necessary grid information for bigelm, and may be used as an 
    input to repeated bigelm runs to avoid recalculation of grid data in each run.
    The Raw_grids and Interpd_grids attributes are instances of the Bigelm_grid
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
                 in which arrays in bigelm will be indexed.
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
    DF_grid = pd.read_csv(grid_file, delimiter=",", header=0, engine="python")
    # The python engine is slower than the C engine, but it works!
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

    # Initialise a custom class instance to hold parameter names and related
    # lists for both the raw and interpolated grids:
    Params = dummy()
    Params.names = grid_params
    Params.n_params = len(grid_params)

    #--------------------------------------------------------------------------
    # Construct raw flux grids
    Raw_grids = construct_raw_grids(DF_grid, Params, lines_list)

    #--------------------------------------------------------------------------
    # Interpolate flux grids
    Interpd_grids = interpolate_flux_arrays(Raw_grids, Params, interpd_grid_shape)

    #--------------------------------------------------------------------------
    # Combine and return results...
    Container1 = dummy()  # Initialise container object
    Container1.Params = Params
    Container1.Raw_grids = Raw_grids
    Container1.Interpd_grids = Interpd_grids

    return Container1



def construct_raw_grids(DF_grid, Params, lines_list):
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
    for p in Params.names:
        # Ensure we have a sorted list of unique values for each parameter:
        param_val_arrs_raw.append( np.sort( np.unique( DF_grid[p].values ) ) )
    # Initialise a grid object to hold the raw grids, arrays of parameter values, etc.:
    Raw_grids = Bigelm_grid( param_val_arrs_raw )

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
    columns = Params.names + lines_list
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
        row_p_vals = ( row_vals[p] for p in Params.names )
        # List the grid indices associated with the param values for this row
        row_p_inds = [Raw_grids.par_indices[j][v] for j,v in enumerate(row_p_vals)]
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



def interpolate_flux_arrays(Raw_grids, Params, interpd_shape):
    """
    Interpolate emission line grids.  Negative values arising from
    interpolation are set to zero (but use of the Akima splines should prevent
    bad "overshoots" in interpolation).

    """
    print("Interpolating model emission line flux grids to shape {0}...".format(tuple(interpd_shape)))

    # Initialise Bigelm_grid object for interpolated arrays
    val_arrs_interp = []
    for p_arr, n in zip(Raw_grids.val_arrs, interpd_shape):
        val_arrs_interp.append( np.linspace( np.min(p_arr), np.max(p_arr), n ) )
    Interpd_grids = Bigelm_grid(val_arrs_interp)

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
        interp_fn = siRGI(tuple(Raw_grids.val_arrs),
                          Raw_grids.grids[emission_line], method="linear")
        # Fill the interpolated fluxes into the final grid structure, using "fancy indexing":
        # Interpd_grids.grids[emission_line][param_fancy_index] = interp_fn( param_combos )
        Interpd_grids.grids[emission_line] = interp_fn( param_combos ).reshape(interpd_shape)
        # I'm gambling here that "reshape" works the way I want... seems like it does!
        # But the fancy indexing seems ot be faster...

    n_lines = len(Interpd_grids.grids)
    line_0, arr_0 = list(Interpd_grids.grids.items())[0]
    print( """Number of bytes in interpolated grid flux arrays: {0} for 1 emission line, 
    {1} total for all {2} lines""".format( arr_0.nbytes, arr_0.nbytes*n_lines,
                                                                      n_lines ) )

    # Set negative values to zero:
    for a in Interpd_grids.grids.values():
        a[a < 0] = 0

    return Interpd_grids



# def interpolate_Akima_nD(data_grid, val_arrs, interpd_shape):
#     """
#     Interpolate data on an n-dimensional rectangular grid to a higher
#     resolution, regular grid using repeated 1D Akima spline interpolation.
#     data_grid: nD numpy array - the input pre-interpolation grid
#     val_arrs: list of 1D numpy arrays - the co-ordinate values corresponding
#             to each of the dimensions in data_grid.  Note
#             that these coordinate values do not need to be uniformly spaced.
#     Returns an n_d numpy array with shape interpd_shape, and a list of arrays
#     specifying the new parameter values along each dimension.  Note that the
#     corner points of the grid remain the same after interpolation.
#     THIS IS TOO SLOW TO BE USABLE AND ALSO INCORRECT, BECAUSE IT IS ONLY
#     A SPLINE IN ONE DIRECTION.
#     """
#     raw_shape = data_grid.shape
#     n_dims = len(raw_shape)
#     assert len(val_arrs) == n_dims and len(interpd_shape) == n_dims
#     assert all(a.size == n for a,n in zip(val_arrs, raw_shape))
#     interpd_shape = tuple(interpd_shape) # Ensure tuple not list
#     val_arrs_out = [np.linspace(a[0],a[-1],n) for a,n in zip(val_arrs,interpd_shape)]

#     intermediate_grid_old = data_grid
#     print("Dimension:", end=" ")
#     for k in range(n_dims): # Iterate over dimensions
#         print(k, end=" ")
#         sys.stdout.flush()
#         # We interpolate the kth dimension
#         intermediate_shape = interpd_shape[:k+1] + raw_shape[k+1:]
#         intermediate_grid_new = np.zeros(intermediate_shape) + np.nan
#         in_par_vals_k, out_par_vals_k = val_arrs[k], val_arrs_out[k]

#         temp_shape = [n for n in intermediate_shape]
#         temp_shape[k] = 1 # Don't include this dimension in Cartesian product
#         # Iterate over all 1D-slices along dimension k
#         for ind_tuple in it_product(*[list(range(n)) for n in temp_shape]):
#             index_list = list(ind_tuple)
#             index_list[k] = slice(None) # slice(None) means ":" (slice along this dimension)
#             y = intermediate_grid_old[index_list].ravel() # 1D array
#             Interpolator = Akima1DInterpolator(in_par_vals_k, y)
#             # Now find the interpolated values for this 1D array
#             intermediate_grid_new[index_list] = Interpolator(out_par_vals_k)

#         intermediate_grid_old = intermediate_grid_new
#     print()

#     interpolated_grid = intermediate_grid_new

#     # Ensure corner points the same!
#     for ind_tuple in it_product(*[(0,-1) for m in range(n_dims)]):
#         assert np.isclose(data_grid[ind_tuple], interpolated_grid[ind_tuple])

#     return interpolated_grid, val_arrs_out



# def interpolate_Akima_2D(data_grid, xvals_in, yvals_in, ninterp_x, ninterp_y):
#     """
#     Interpolate a 2D grid to a higher resolution, regular grid using repeated 1D
#     Akima spline interpolation.
#     data_grid: 2D numpy array - the input grid
#     xvals_in, yvals_in: 1D numpy arrays - the co-ordinate values corresponding
#         to the 0th and 1st dimensions respectively in data_grid.  Note
#         that these coordinate values do not need to be uniformly spaced.
#     Returns a 2d numpy array with shape (ninterp_x, ninterp_y).  The corner
#     points of the grid remain the same.
#     This function isn't used except in testing interpolate_Akima_nD.
#     THIS IS INCORRECT BECAUSE IT ISN'T A PROPER 2D SPLINE.
#     """
#     nx_in, ny_in = data_grid.shape  # Indexing is y first, then x.
#     assert xvals_in.size == nx_in and yvals_in.size == ny_in
#     xvals_out = np.linspace(xvals_in[0], xvals_in[-1], ninterp_x)
#     yvals_out = np.linspace(yvals_in[0], yvals_in[-1], ninterp_y)
#     interpolated_grid = np.zeros((ninterp_x, ninterp_y)) + np.nan
#     intermediate_grid = np.zeros((nx_in, ninterp_y)) + np.nan
#     # The intermediate grid will have interpolated rows, which are at the
#     # x-positions of the rows in the input grid, not at the x-positions of the
#     # interpolated grid.
#     for i in range(nx_in):
#         # For each x-value fit an Akima spline to the corresponding y-values (row)
#         Col_interpolator = Akima1DInterpolator(yvals_in, data_grid[i,:])
#         # Now find the interpolated y-values for this row
#         intermediate_grid[i,:] = Col_interpolator(yvals_out)

#     for j in range(ninterp_y):
#         # For each y-value fit an Akima spline to the corresponding x-values (column)
#         Row_interpolator = Akima1DInterpolator(xvals_in, intermediate_grid[:,j])
#         # Now fill in the output grid for this columns
#         interpolated_grid[:,j] = Row_interpolator(xvals_out)

#     for i,j in [(0,0), (0,-1), (-1,0), (-1,-1)]: # Ensure corner points the same!
#         assert np.isclose(data_grid[i,j], interpolated_grid[i,j])

#     return interpolated_grid



# def setup_interpolators(Raw_grids, interp="Linear"):
#     """
#     Raw_grids: Contains details of the input model grid
#     interp: can be "Linear", "Spline"
#     Returns a dictionary mapping emission line names to callable "interpolators"
#     """

#     # Some preparation for using the "Spline" method
#     p_index_interpolators = []
#     for a in Raw_grids.val_arrs: # Setup linear interpolator for each param
#         interpolator = interp1d(x=a, y=np.arange(len(a)), bounds_error=False,
#                                 fill_value=(0,len(a)-1))
#         p_index_interpolators.append(interpolator)
#     # So p_index_interpolators[j][f] will give the interpolated "index"
#     # corresponding to the value f along the j axis (i.e. to parameter j having
#     # value f).  We're converting the actual value of the parameter to a "pixel"
#     # coordinate.

#     # Iterate over emission lines, storing an "interpolator" for each:
#     line_interpolators = {} # Keys are emisison line names
#     for line, flux_grid in Raw_grids.grids.items():

#         # LinearGridInterpolator = RegularGridInterpolator(Raw_grids.val_arrs,
#         #     flux_grid, method="linear", bounds_error=False, fill_value=None)
#         # LinearGridInterpolator = RegularGridInterpolator(Raw_grids.val_arrs,
#         #     flux_grid, method="linear", bounds_error=False, fill_value=1e55) # Return massive flux if outside range!
#         LinearGridInterpolator = RegularGridInterpolator(Raw_grids.val_arrs,
#             flux_grid, method="linear", bounds_error=True, fill_value=1e55) # Return massive flux if outside range!
#         # def line_interpolator_linear(p_vector):
#         #     """
#         #     A wrapper around RegularGridInterpolator - necessary because I 
#         #     don't like the options for values outside the parameter space.
#         #     """
#         #     # Move values outside the bounds onto the boundaries
#         #     p_new = []
#         #     for i,(p,(p_min,p_max)) in enumerate(zip(p_vector, Raw_grids.p_minmax)):
#         #         p_new.append( max(p_min, min(p, p_max)) )

#         #     return LinearGridInterpolator( p_new )


#         def line_interpolator_spline(p_vector):
#             """
#             Function for interpolating the value of an emission line, given a
#             list of parameter values.
#             """
#             # http://stackoverflow.com/questions/6238250/multivariate-spline-interpolation-in-python-scipy
#             # Firstly convert the parameter values to pixel coordinates:
#             coords = [p_index_interpolators[i](f) for i,f in enumerate(p_vector)]
#             coords = np.array(coords)[:,np.newaxis] # Need transpose
#             # Now interpolate in the flux grid using the pixel coordinates:
#             flux = ndimage.map_coordinates(flux_grid, coords, order=3)
#             return flux[0] # Return as a float, not a numpy array
#             # In the coords array, each column is a point to be interpolated/
#             # Here we have only one column.
#             # We use 3rd-order (cubic) spline interpolation.
#             # Interpolating outside the grid will just give "nearest" grid
#             # edge values

#             # Lingering questions:
#             # - Could there be an issue with the spline interpolation
#             # returning negative flux values?
#             # - Is this too slow because the array is being copied in memory
#             # on each interpolation?
#             # - Does this naive spline interpolation (not Akima spline) behave
#             # poorly with "outliers" in the model grid data?

#         if interp == "Linear":
#             line_interpolators[line] = LinearGridInterpolator#line_interpolator_linear
#         elif interp == "Spline":
#             line_interpolators[line] = line_interpolator_spline
#         else:
#             raise ValueError("Unknown value given for keyword 'interp'")

#     return line_interpolators


