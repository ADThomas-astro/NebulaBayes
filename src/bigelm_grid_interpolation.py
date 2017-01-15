from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt
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

    container1 = Bigelm_container()  # Initialise container object

    # Load ASCII database csv table containing the model grid output:
    D_table = pd.read_csv( grid_file, delimiter=",", header=0, engine="python" )
    # The python engine is slower than the C engine, but it works!
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
        D_table[line][ ~np.isfinite( D_table[line].values ) ] = 0
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
    # Set up raw grid and interpolation...

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

    # # Determine the parameter values for the interpolated grid,
    # # and then initialise a grid object for the interpolated grid:
    # # Create a list of arrays, where each array contains a list of constantly-spaced values for
    # # the corresponding parameter in the interpolated grid:
    # param_val_arrs_interp = []
    # for p_arr, n in zip(Raw_grids.val_arrs, interpd_grid_shape):
    #     param_val_arrs_interp.append( np.linspace( np.min(p_arr),
    #                                                np.max(p_arr), n ) )
    # # Initialise Bigelm_grid object to hold the interpolated grids, arrays of parameter values, etc.:
    # Interpd_grids = Bigelm_grid( param_val_arrs_interp )
    # # List of interpolated grid spacing for each parameter
    # # (needed for integration and plotting later):
    # Interpd_grids.spacing = [ (val_arr[1] - val_arr[0]) for val_arr in Interpd_grids.val_arrs ]

    #--------------------------------------------------------------------------
    # Construct the raw model grids as multidimensional arrays...

    print("Building flux arrays for the model grids...")
    # How long does this take?
    # We use an inefficient method for building the model grids because we're
    # not assuming anything about the order of the rows in the input table.
    # First reduce D_table to include only the required columns:
    D_table2 = D_table[ (Params.names + lines_list) ]
    for emission_line in lines_list:
        # Initialise new (emission_line, flux_array) item in dictionary, as an array of nans:
        Raw_grids.grids[emission_line] = np.zeros( Raw_grids.shape ) + np.nan
        # Populate the model flux array for this emission line:
        for row, flux in enumerate( D_table[emission_line] ): # Iterate column
            # Generator for the value of each grid parameter in this row
            row_p_vals = ( D_table.ix[row,p] for p in Params.names )
            # Find the set of grid indices associated with the parameter values:
            p_indices = [Raw_grids.val_indices[j][v] for j,v in enumerate(row_p_vals)]
            # Write the flux value for this gridpoint into the correct location
            # in the grid array for this emission line:
            Raw_grids.grids[emission_line][tuple(p_indices)] = flux

    arr_n_bytes = Raw_grids.grids[lines_list[0]].nbytes
    n_lines = len(lines_list)
    print( """Number of bytes in raw grid flux arrays: {0} for 1 emission line, 
    {1} total for all {2} lines""".format( arr_n_bytes, arr_n_bytes*n_lines,
                                                                    n_lines ) )

    import sys
    sys.exit()

    #--------------------------------------------------------------------------
    # Interpolate model grids to a higher resolution and even spacing...

    print("Interpolating flux arrays for the model grids...")
    # # # A list of all parameter value combinations in the
    # # # interpolated grid in the form of a numpy array:
    # # param_combos = np.array( list(
    # #         itertools.product( *param_val_arrs_interp ) ) )
    # # # A list of all index combinations for the
    # # # interpolated grid, corresponding to param_combos:
    # # param_index_combos = np.array( list(
    # #         itertools.product( *[np.arange(n) for n in Interpd_grids.shape] ) ) )
    # Generate a tuple containing an index array for each grid dimension,
    # with each combination of indices (e.g. from the 7th entry in each
    # index array) correspond to a param_combos row (e.g. the 7th):
    
    # param_fancy_index = tuple(
    #         [ param_index_combos[:,i] for i in np.arange(Params.n_params) ] )
    # This will be used to take advantage of numpy's fancy indexing capability.

    for emission_line in lines_list:
        pass
        # # Create new (emission_line, flux_array) item in dictionary of interpolated grid arrays:
        # Interpd_grids.grids[emission_line] = np.zeros( Interpd_grids.shape )
        

        # Function here!
        # # Create function for carrying out the interpolation:
        # interp_fn = siRGI(tuple(Raw_grids.val_arrs),
        #                   Raw_grids.grids[emission_line], method="linear")
        
    #     # Fill the interpolated fluxes into the final grid structure, using "fancy indexing":
    #     Interpd_grids.grids[emission_line][param_fancy_index] = interp_fn( param_combos )
    # print( "An interpolated grid flux array for a single emission line is " + 
    #        str(Interpd_grids.grids[lines_list[0]].nbytes) + " bytes" )

    #--------------------------------------------------------------------------
    # Combine and return results...

    container1.Params        = Params
    container1.Raw_grids     = Raw_grids
    # container1.Interpd_grids = Interpd_grids

    return container1






    # http://stackoverflow.com/questions/6238250/multivariate-spline-interpolation-in-python-scipy

    # Note that the output interpolated coords will be the same dtype as your input
    # data.  If we have an array of ints, and we want floating point precision in
    # the output interpolated points, we need to cast the array as floats
    data = np.arange(40).reshape((8,5)).astype(np.float)

    # I'm writing these as row, column pairs for clarity...
    coords = np.array([[1.2, 3.5], [6.7, 2.5], [7.9, 3.5], [3.5, 3.5]])
    # However, map_coordinates expects the transpose of this
    coords = coords.T

    # The "mode" kwarg here just controls how the boundaries are treated
    # mode='nearest' is _not_ nearest neighbor interpolation, it just uses the
    # value of the nearest cell if the point lies outside the grid.  The default is
    # to treat the values outside the grid as zero, which can cause some edge
    # effects if you're interpolating points near the edge
    # The "order" kwarg controls the order of the splines used. The default is 
    # cubic splines, order=3
    zi = ndimage.map_coordinates(data, coords, order=3, mode='nearest')

    row, column = coords
    nrows, ncols = data.shape
    im = plt.imshow(data, interpolation='nearest', extent=[0, ncols, nrows, 0])
    plt.colorbar(im)
    plt.scatter(column, row, c=zi, vmin=data.min(), vmax=data.max())
    for r, c, z in zip(row, column, zi):
        plt.annotate('%0.3f' % z, (c,r), xytext=(-10,10), textcoords='offset points',
                arrowprops=dict(arrowstyle='->'), ha='right')
    plt.show()

