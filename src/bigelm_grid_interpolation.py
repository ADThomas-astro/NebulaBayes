from __future__ import print_function, division
import numpy as np
import pandas as pd
# from scipy import ndimage
# from scipy.interpolate import interp1d
from scipy.interpolate import Akima1DInterpolator#, RectBivariateSpline
from .bigelm_classes import Bigelm_container, Grid_parameters, Bigelm_grid






#============================================================================
def initialise_grids(grid_file, grid_params, lines_list):
    """
    Initialise grids and return a Bigelm_container object, which will have
    attributes Params, Raw_grids.  # and Interpd_grids.
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
    # interpd_grid_shape: A tuple of integers, giving the size of each dimension
    #                     of the interpolated grid.  The order of the integers
    #                     corresponds to the order of parameters in grid_params.
    #                     The default is 30 gridpoints along each dimension.  Note
    #                     that the number of interpolated gridpoints entered in
    #                     interpd_grid_shape may have a major impact on the speed
    #                     of the program.
    """
    print("Loading input grid table...")
    # Load database csv table containing the model grid output
    D_table = pd.read_csv( grid_file, delimiter=",", header=0, engine="python" )
    # The python engine is slower than the C engine, but it works!
    print("Cleaning input grid table...")
    # Remove any whitespace from column names
    D_table.rename(inplace=True, columns={c:c.strip() for c in D_table.columns})
    for line in lines_list: # Ensure line columns have float dtype
        D_table[line] = pd.to_numeric(D_table[line], errors="coerce")
        # We "coerce" errors to NaNs, which will be set to zero

    # Clean and check the model data:
    for line in lines_list:
        # Check that all emission lines in input are also in the model data:
        if not line in D_table.columns:
            raise ValueError("Measured emission line " + line +
                             " was not found in the model data.")
        # Set any non-finite model fluxes to zero:
        ####### Dodgy?!?!
        D_table.loc[~np.isfinite(D_table[line].values), line] = 0
        # pandas complains about D_table[line][ ~np.isfinite( D_table[line].values ) ] = 0
        # Check that all model flux values are non-negative:
        if np.sum( D_table[line].values < 0 ) != 0:
            raise ValueError("A flux value for modelled emission line " +
                             line + " is negative.")

    # Initialise a custom class instance to hold parameter names and related
    # lists for both the raw and interpolated grids:
    Params = Grid_parameters( grid_params )

    # # Number of points for each parameter in interpolated grid, in order:
    # if interpd_grid_shape == None: # If not specified by user:
    #     interpd_grid_shape = tuple( [30]*Params.n_params ) # Default
    # else: # If specified by user, check that it's the right length:
    #     if len(interpd_grid_shape) != Params.n_params:
    #         raise ValueError("interpd_grid_shape should contain" + 
    #                          "exactly one integer for each parameter" )

    #--------------------------------------------------------------------------
    # Set up raw grid...

    # Determine the list of parameter values for the raw grid:
    # Initialise list of arrays, with each array being the list of grid values for a parameter:
    param_val_arrs_raw = []
    for p in Params.names:
        # Ensure we have a sorted list of unique values for each parameter:
        param_val_arrs_raw.append( np.sort( np.unique( D_table[p].values ) ) )
    # Initialise a grid object to hold the raw grids, arrays of parameter values, etc.:
    Raw_grids = Bigelm_grid( param_val_arrs_raw )

    # Check that the input database table is the right length:
    # (This is equivalent to checking that we have a regular grid, e.g. without missing values)
    if Raw_grids.n_gridpoints != len(D_table):
        raise ValueError("The input model grid table does not" + 
                         "have a consistent length.")

    #--------------------------------------------------------------------------
    # Construct the raw model grids as a multidimensional array for each line
    print("Building flux arrays for the model grids...")
    # We use an inefficient method for building the model grids because we're
    # not assuming anything about the order of the rows in the input table.
    # First reduce D_table to include only the required columns:
    columns = Params.names + lines_list
    D_table = D_table[ columns ]
    for emission_line in lines_list: # Initialise new (emission_line,flux_array)
        # item in dictionary, as an array of nans:
        Raw_grids.grids[emission_line] = np.zeros( Raw_grids.shape ) + np.nan
    # Iterate over rows in the input model grid table:
    for row_tuple in D_table.itertuples(index=False, name=None):
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

    #--------------------------------------------------------------------------
    # Combine and return results...
    container1 = Bigelm_container()  # Initialise container object
    container1.Params = Params
    container1.Raw_grids = Raw_grids
    # container1.flux_interpolators = setup_interpolators(Raw_grids)

    return container1



#============================================================================
def interpolate_Akima_2D(data_grid, xvals_in, yvals_in, ninterp_x, ninterp_y):
    """
    Interpolate a 2D grid to a higher resolution, regular grid using repeated 1D
    Akima spline interpolation.
    data_grid: 2D numpy array - the input grid
    xvals_in, yvals_in: 1D numpy arrays - the co-ordinate values corresponding
        to the 0th and 1st dimensions respectively in data_grid.  Note
        that these coordinate values do not need to be uniformly spaced.
    Returns a 2d numpy array with shape (ninterp_x, ninterp_y).  The corner
    points of the grid remain the same.
    """
    nx_in, ny_in = data_grid.shape  # Indexing is y first, then x.
    assert xvals_in.size == nx_in and yvals_in.size == ny_in
    xvals_out = np.linspace(xvals_in[0], xvals_in[-1], ninterp_x)
    yvals_out = np.linspace(yvals_in[0], yvals_in[-1], ninterp_y)
    interpolated_grid = np.zeros((ninterp_x, ninterp_y)) + np.nan
    intermediate_grid = np.zeros((nx_in, ninterp_y)) + np.nan
    # The intermediate grid will have interpolated rows, which are at the
    # x-positions of the rows in the input grid, not at the x-positions of the
    # interpolated grid.
    for i in range(nx_in):
        # For each x-value fit an Akima spline to the corresponding y-values (row)
        Col_interpolator = Akima1DInterpolator(yvals_in, data_grid[i,:])
        # Now find the interpolated y-values for this row
        intermediate_grid[i,:] = Col_interpolator(yvals_out)

    for j in range(ninterp_y):
        # For each y-value fit an Akima spline to the corresponding x-values (column)
        Row_interpolator = Akima1DInterpolator(xvals_in, intermediate_grid[:,j])
        # Now fill in the output grid for this columns
        interpolated_grid[:,j] = Row_interpolator(xvals_out)

    for i,j in [(0,0), (0,-1), (-1,0), (-1,-1)]: # Ensure corner points the same!
        assert np.isclose(data_grid[i,j], interpolated_grid[i,j])

    return interpolated_grid



#============================================================================
def interpolate_posteriors(Raw_grids, Params, posteriors_1D, posteriors_2D, n_interp_pts):
    """
    Interpolate marginalised posteriors.  Negative values arising from
    interpolation are set to zero (but use of the Akima splins should prevent
    bad "overshoots" in interpolation).

    """
    val_arrs_dict = dict(zip(Params.names, Raw_grids.val_arrs))
    # val_arrs_dict maps parameter names to an array of values for that parameter
    # along the corresponding dimension of the raw grid
    # Determine the parameter values for the interpolated posteriors:
    param_val_arrs_interp = {}
    for p1 in posteriors_1D:
        val_arr1 = val_arrs_dict[p1]
        param_val_arrs_interp[p1] = np.linspace(np.min(val_arr1),
                                                  np.max(val_arr1), n_interp_pts)

    posteriors_1D_interp, posteriors_2D_interp = {}, {}
    for p1, posterior in posteriors_1D.items():
        in_arr = val_arrs_dict[p1] # Pre-interpolation parameter values
        Interpolator = Akima1DInterpolator(x=in_arr, y=posterior)
        out_arr = param_val_arrs_interp[p1] # Interpolated parameter values
        posteriors_1D_interp[p1] = Interpolator(out_arr)
    for (p_x, p_y), posterior in posteriors_2D.items():
        # Now p_x, p_y correspond to axes 0 and 1 of the posterior; note that
        # when the posterior is displayed, axis 0 (p_x) is the vertical axis.
        in_arr_1, in_arr_2 = val_arrs_dict[p_x], val_arrs_dict[p_y] # Pre-interpolation parameter values
        # # Linear interpolation:
        # Interpolator = RectBivariateSpline(x=in_arr_1, y=in_arr_2, z=posterior,
        #                                    kx=1, ky=1) # Linear in x and y
        # # Here x- and y- refer to axes 0 and 1.
        # out_arr_1 = param_val_arrs_interp[p_x] # Interpolated parameter values
        # out_arr_2 = param_val_arrs_interp[p_y] # Interpolated parameter values
        # posteriors_2D_interp[(p_x, p_y)] = Interpolator(x=out_arr_1, y=out_arr_2)
        # 2D Akima spline interpolation:
        posteriors_2D_interp[(p_x, p_y)] = interpolate_Akima_2D(posterior,
                                xvals_in=in_arr_1, yvals_in=in_arr_2,
                                ninterp_x=n_interp_pts, ninterp_y=n_interp_pts)

    # Set negative values to zero:
    for p in [posteriors_1D_interp, posteriors_2D_interp]:
        for a in p.values():
            a[a < 0] = 0

    return posteriors_1D_interp, posteriors_2D_interp, param_val_arrs_interp



# #============================================================================
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


