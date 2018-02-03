from __future__ import print_function, division
from collections import OrderedDict as OD
import itertools  # For Cartesian product
import logging
import os  # For path manipulations

from astropy.io import fits  # For reading FITS binary tables
from astropy.table import Table  # For FITS table to pandas DataFrame conversion
import numpy as np  # Core numerical library
import pandas as pd # For tables ("DataFrame"s)
from scipy.ndimage import map_coordinates  # For spline interpolation in 3D
from ._compat import _str_type  # Compatibility

# Directory of built-in grids
GRIDS_LOCATION = os.path.join(os.path.dirname(__file__), "grids")


"""
This module contains code to load the model grid database table, constuct
model flux arrays, and interpolate those arrays to higher resolution.

Adam D. Thomas 2015 - 2018
"""


NB_logger = logging.getLogger("NebulaBayes")



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
        param_value_arrs: List of 1D arrays of parameter values over the grid.
                          The list ordering corresponds to param_names.
        Note that NebulaBayes code relies on the dictionaries below being
        ordered.
        """
        assert len(param_names) == len(param_value_arrs)
        # Record some basic info
        self.param_names = param_names
        self.param_values_arrs = param_value_arrs
        self.ndim = len(param_names)
        self.shape = tuple([len(arr) for arr in param_value_arrs ])
        self.n_gridpoints = np.product( self.shape )

        # Define mappings for easily extracting data about the grid
        self.paramName2ind = OD(zip(param_names, range(self.ndim)))
        self.paramName2paramValueArr = OD(zip(param_names, param_value_arrs))

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
    Will hold a grid of the same shape for each emission line.
    """
    def __init__(self, param_names, param_value_arrs):
        """
        Initialise an NB_Grid instance.  The attributes describing the grid are
        stored as attributes both on the instance and in the attribute
        "_Grid_spec".
        """
        super(NB_Grid, self).__init__(param_names, param_value_arrs)
        self._Grid_spec = Grid_description(param_names, param_value_arrs)
        self.grids = OD()  # We rely on this being ordered
        # For the raw grids, this "grids" dict holds arrays under the line name
        # directly.  For the interpolated grids, the "grids" attribute holds
        # other dicts, named e.g. "No_norm" and "Hbeta_norm", corresponding to
        # different normalisations.  When we normalise we may lose information
        # (where the normalising grid has value zero), so we need multiple
        # copies of the interpolated grids for different normalisations.  Every
        # time we want a new normalisation, we add another dict (set of grids)
        # to the "grids" dict.



def initialise_grids(grid_table, grid_params, lines_list, interpd_grid_shape,
                     interp_order):
    """
    Initialise grids objects for an initialising NB_Model instance.  The
    outputs are instances of the NB_Grid class defined above.
    
    Parameters
    ----------
    grid_table : str or pandas DataFrame
        The table of photoionisation model grid fluxes, given as the
        filename of a csv, FITS (.fits) or compressed FITS (fits.gz) file,
        a pandas DataFrame instance, or one of the strings "HII" or "NLR".
    grid_params : list of strings
        The names of the grid parameters.
    lines_list : list of strings or None
        The emission lines to be interpolated from the raw flux grids
    interpd_grid_shape : tuple of integers
        The size of each dimension of the interpolated flux grids.
    interp_order : integer (1 or 3)
        The order of the polynomials to use for interpolation.

    Returns
    -------
    Raw_grids : NB_Grid instance
    Interpd_grids : NB_Grid instance
    """
    # Load database table containing the model grid output
    DF_grid = load_grid_data(grid_table)
    # Process and check the table, making the lines_list if it wasn't specified
    DF_grid, lines_list = process_raw_table(DF_grid, grid_params, lines_list)
    
    # Construct raw flux grids
    Raw_grids = construct_raw_grids(DF_grid, grid_params, lines_list)

    # Interpolate flux grids
    Interpd_grids = interpolate_flux_arrays(Raw_grids, interpd_grid_shape,
                                            interp_order=interp_order)

    return Raw_grids, Interpd_grids



def load_grid_data(grid_table):
    """
    Load the model grid data.
    
    Returns
    -------
    DF_grid : pd.DataFrame instance
    """
    NB_logger.info("Loading input grid data...")

    if isinstance(grid_table, _str_type):
        if grid_table in ["HII", "NLR"]:
            # Use a built-in grid.  Resolve shorthand string to the full path.
            grid_name = "NB_{0}_grid.fits.gz".format(grid_table)
            grid_table = os.path.join(GRIDS_LOCATION, grid_name)

        if grid_table.endswith((".fits", ".fits.gz")):
            # This includes the built-in grids
            BinTableHDU_0 = fits.getdata(grid_table, 0)
            DF_grid = Table(BinTableHDU_0).to_pandas()
        elif grid_table.endswith(".csv"):
            DF_grid = pd.read_table(grid_table, header=0, delimiter=",")
        else:
            if "." in grid_table:
                raise ValueError("grid_table has unknown file extension")
            else:
                raise ValueError("Unknown grid_table string '{0}'".format(
                                                                   grid_table))
    elif isinstance(grid_table, pd.DataFrame):
        # Copy the table, so we don't surprise the user when we modify it!
        DF_grid = grid_table.copy()
    else:
        raise TypeError("grid_table should be a string or DataFrame, not a " +
                        str(type(grid_table)))

    return DF_grid



def process_raw_table(DF_grid, grid_params, lines_list):
    """
    Ensure grid data are double-precision, finite and non-negative.
    Check that the grid_params are all found in the table header, and check
    that the supplied lines are in the table header.  If the lines_list wasn't
    specified we create it.

    Returns
    -------
    DF_grid : pd.DataFrame instance
    lines_list : List of strings.  The names of the lines to be interpolated.
    """
    if len(DF_grid) == 0:
        raise ValueError("Input model grid table contains no rows")

    # Remove any whitespace from column names
    DF_grid.rename(inplace=True, columns={c:c.strip() for c in DF_grid.columns})

    for p in grid_params:
        if p not in DF_grid.columns:
            raise ValueError("Grid parameter {0} not found in grid".format(p))

    if lines_list is None:  # Make lines_list if not specified by user
        lines_list = [l for l in DF_grid.columns if l not in grid_params]

    # Clean and check the model data:
    for line in lines_list:
        if line not in DF_grid.columns:
            raise ValueError("Emission line {0} not found in grid".format(line))
        # Set any non-finite model fluxes to zero.  Is this the wrong thing to
        # do?  It's documented at least, in NB0_Main.py.
        DF_grid.loc[~np.isfinite(DF_grid[line].values), line] = 0
        # Check that all model flux values are non-negative:
        if np.any(DF_grid[line].values < 0):
            raise ValueError("A model flux value for emission line " + line +
                             " is negative.")
        if np.all(DF_grid[line].values == 0):
            NB_logger.warning("WARNING: All model fluxes for emission line "
                              "{0} are zero.".format(line))

        # Ensure line flux columns are a numeric data type
        DF_grid[line] = pd.to_numeric(DF_grid[line], errors="raise")
        DF_grid[line] = DF_grid[line].astype("float64") # Ensure double precision

    return DF_grid, lines_list



def construct_raw_grids(DF_grid, grid_params, lines_list):
    """
    Construct arrays of flux grids from the input flux table.

    Parameters
    ----------
    DF_grid: pandas DataFrame
        Table holding the predicted fluxes of the model grid.
    grid_params:  list of str
        The names of the grid parameters, matching columns in DF_grid.
    lines_list: list of str
        The names of emission lines of interest, matching columns in DF_grid.
    """
    # Determine the list of parameter values for the raw grid:
    # List of arrays; each array holds the grid values for a parameter:
    param_val_arrs_raw = []
    for p in grid_params:
        # Ensure we have a sorted list of unique values for each parameter:
        p_arr = np.sort(np.unique(DF_grid[p].values))
        n_p = p_arr.size
        if n_p < 3:
            raise ValueError("At least 3 unique values are required for each "
                  "grid parameter. '{0}' has only {1} value(s)".format(p, n_p))
        param_val_arrs_raw.append(p_arr)
    # Initialise a grid object to hold the raw grids:
    Raw_grids = NB_Grid(grid_params, param_val_arrs_raw)

    # Check that the input database table is the right length:
    # (This is equivalent to checking that we have a rectangular grid, e.g.
    # without missing values.  The spacing does not need to be uniform.)
    if Raw_grids.n_gridpoints != len(DF_grid):
        raise ValueError("There must be exactly one gridpoint for every " +
                "possible combination of included parameter values.  The " +
                "input model grid table length of {0} ".format(len(DF_grid)) +
                "is not equal to the product of the number of values of " +
                "each parameter ({0}).".format(Raw_grids.n_gridpoints) +
                "\nParameter values:\n" +
                "\n".join("{0} ({1} values): {2}".format(p, len(vals), vals)
                           for p, vals in zip(grid_params, param_val_arrs_raw)))

    #--------------------------------------------------------------------------
    # Construct the raw model grids as a multidimensional array for each line
    NB_logger.info("Building flux arrays for the model grids...")
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

    return Raw_grids



def interpolate_flux_arrays(Raw_grids, interpd_shape, interp_order):
    """
    Interpolate emission line grids, using linear or cubic interpolation, and
    in arbitrary dimensions.

    Parameters
    ----------
    Raw_grids : object
        Object holding the raw (uninterpolated) model flux grids
    interpd_shape : tuple of integers
        The size of each dimension of the interpolated flux grids.
    interp_order : integer (1 or 3)
        The order of the polynomials to use for interpolation.

    Notes
    -----
    We do not normalise the grid fluxes to the norm_line fluxes here (they're
    normalised just before calculating the likelihood), and we store the
    interpolated grids under the name "No_norm".
    Note that we require that the spacing in the interpolated grids is uniform,
    becuase we'll be assuming this when integrating to marginalise PDFs, and
    also we'll be using matplotlib.pyplot.imshow to show an image of PDFs on
    the interpolated array, and imshow (as you would expect) assumes
    "evenly-spaced" pixels.
    """
    interp_types = {1: "linear", 3: "cubic"}
    interp_type = interp_types[interp_order]
    NB_logger.info("Resampling model emission line flux grids to shape {0} "
        "using {1} interpolation...".format(tuple(interpd_shape), interp_type))

    # Initialise NB_Grid object for interpolated arrays
    # First find the interpolated values of the grid parameters
    val_arrs_interp = []
    for i, (p, n) in enumerate(zip(Raw_grids.param_names, interpd_shape)):
        p_min, p_max = Raw_grids.paramName2paramMinMax[p]
        val_arrs_interp.append(np.linspace(p_min, p_max, n))
    
    Interpd_grids = NB_Grid(list(Raw_grids.param_names), val_arrs_interp)
    Interpd_grids.grids["No_norm"] = OD()
    
    # Check that the interpolated grid has uniform spacing in each dimension:
    for arr in Interpd_grids.param_values_arrs:
        arr_diff = np.diff(arr)
        assert np.allclose(arr_diff, arr_diff[0])

    if interp_order == 1:  # Create class for carrying out the interpolation:
        Interpolator = RegularGridResampler(Raw_grids.param_values_arrs,
                                            Interpd_grids.shape)

    # Iterate emission lines, doing the interpolation:
    for emission_line, raw_flux_arr in Raw_grids.grids.items():
        NB_logger.info("    Interpolating for {0}...".format(emission_line))
        if interp_order == 1:
            interp_vals, interp_arr = Interpolator(raw_flux_arr)
            for a1, a2 in zip(interp_vals, Interpd_grids.param_values_arrs):
                assert np.array_equal(a1, a2)
        else:  # interp_order == 3
            interp_arr = resample_grid_with_cubic_splines(raw_flux_arr,
                                    Raw_grids.param_values_arrs, interpd_shape)
        assert np.all(np.isfinite(interp_arr))
        Interpd_grids.grids["No_norm"][emission_line] = interp_arr


    n_lines = len(Interpd_grids.grids["No_norm"])
    line_0, arr_0 = list(Interpd_grids.grids["No_norm"].items())[0]
    arr_MB = arr_0.nbytes / 1e6
    NB_logger.debug("Interpolated flux grid size is {0:.2f}MB for 1"
                    " line and {0:.2f}MB total for all {1} lines".format(
                                            arr_MB, arr_MB*n_lines, n_lines))

    # Set negative values to zero: (there shouldn't be any if we're using
    # linear interpolation)
    for a in Interpd_grids.grids["No_norm"].values():
        np.clip(a, 0., None, out=a)

    # Store the interpolation polynomial order for potential later reference
    Interpd_grids.interp_order = interp_order

    return Interpd_grids



def cartesian_prod(arrays, out=None):
    """
    Generate a cartesian product of input arrays recursively.
    Copied from:
    https://stackoverflow.com/questions/1208118/
            using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    https://stackoverflow.com/questions/28684492/
                                 numpy-equivalent-of-itertools-product?rq=1
    This method is much faster than constructing a numpy array using
    itertools.product().

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Example
    -------
    >>> cartesian_prod(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros((n, len(arrays)), dtype=dtype)

    m = n // arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian_prod(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]

    return out



def resample_grid_with_cubic_splines(raw_arr, param_val_lists, out_shape):
    """
    Resample a rectangular n-dimensional grid (with uneven spacing along one or
    more dimensions) to a different (regular) sampling, using (approximately)
    cubic spline interpolation.

    Parameters
    ----------
    raw_arr : numpy ndarray
        The input array to resample
    param_val_lists : list of lists of floats
        The input parameter values along each dimension (in order)
    out_shape : tuple of ints
        The output shape, which specifies the number of evenly-spaced sample
        points along each dimension

    Notes
    -----
    The technique used here isn't strictly mathematically correct. We use the
    scipy.ndimage.map_coordinates function, which assumes that the input array
    is an image with even spacing between the pixels.  This is not the case,
    but as a fudge we calculate the output coordinates in pixel space taking
    into account the uneven input sampling.  Hence the results should be
    resonable, although the actual spline functions used in the interpolation
    will presumably not have quite the correct shape.  The differences will be
    larger for input arrays with more uneven sampling.
    """
    # Output parameter values
    out_param_vals = []
    for p_vals_i, n_i in zip(param_val_lists, out_shape): # Iterate dimensions:
        # Evenly sampled points in this dimension
        out_param_vals.append(np.linspace(p_vals_i[0], p_vals_i[-1], n_i))

    # For the interpolation routine and method we're using, we need the
    # input points for the interpolation to be in pixel coordinates.  The
    # input parameter spacing is not uniform, so we need to calculate the
    # locations of the output parameter values in "input index" coordinates.
    index_locations = [[] for _ in out_shape]
    # Iterate over dimensions
    for i, (in_vals_i, out_vals_i) in enumerate(zip(param_val_lists,
                                                    out_param_vals)):
        # Iterate over sample points we'll be interpolating to
        for x_j in out_vals_i:
            closest_below_ind = (np.searchsorted(np.array(in_vals_i), x_j,
                                                 side="right") - 1)
            # closest_below_ind is the index of the raw grid parameter value
            # that is closest to x_j without being larger (it's possibly equal)
            if closest_below_ind < len(in_vals_i) - 1:
                below_val = in_vals_i[closest_below_ind]
                width = in_vals_i[closest_below_ind+1] - below_val
                x_j_ind = closest_below_ind + (x_j - below_val) / width
            else:  # At maximum of parameter space
                x_j_ind = closest_below_ind
            index_locations[i].append(x_j_ind)
        assert len(index_locations[i]) == out_shape[i]

    # Make a list of points to interpolate, which includes every point in the
    # interpolated grid.  There is a row for each dimension, and a column for
    # each point in the interpolated grid.
    coords = cartesian_prod(index_locations).T

    # Perform the interpolation
    interped = map_coordinates(raw_arr, coordinates=coords, order=3)
    out_arr = interped.reshape(out_shape)

    # # Sanity check - ensure the values are unchanged at all corners of the grid
    # for corner_ind_tuple in itertools.product(*([[0, -1]] * len(out_shape))):
    #     if not np.isclose(raw_arr[corner_ind_tuple], out_arr[corner_ind_tuple],
    #                       atol=1e-8, rtol=0):  # Note - normalised to Hbeta
    #         raise ValueError("Corner {0}: Raw ({1}) doesn't equal interpolated ({2})".format(
    #             corner_ind_tuple, raw_arr[corner_ind_tuple], out_arr[corner_ind_tuple]))

    return out_arr



class RegularGridResampler(object):
    """
    Interpolate a regular grid in arbitrary dimensions to uniform sampling
    in each dimension ("re-grid the data"), potentially to a higher resolution.
    Linear interpolation is used.

    The RegularGridResampler is initialised with an input grid shape and an
    output grid shape, to be ready to interpolate from the input shape to the 
    output shape.  Each call then provides different grid data to be
    interpolated; this code is optimised for doing the same interpolation
    on many different grids of the same shape.

    The input grid data must be defined on a regular grid, but the grid spacing
    may be uneven.  The output grid will have even spacing, and the "corner"
    gridpoints and values will be the same as in the input grid.

    Parameters
    ----------
    in_points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.

    out_shape : tuple of ints
        The number of evenly spaced interpolated points in each dimension for
        output interpolated grids

    Notes
    -----
    Based on the same method as is used in the scipy RegularGridInterpolator,
    which itself is based on code by Johannes Buchner, see
    https://github.com/JohannesBuchner/regulargrid

    The method is as follows (consider just one interpolation point for now):
    Iterate over "edges", which are the 2**ndim points around the interpolation
    point that are relevant to the interpolation.  An "edge" is on the "lower"
    or "upper" side of the interpolated point in a given dimension.  Each of the
    2**ndim "edges" contributes to the interpolated value.
    For each "edge", find the total weight.  The weight for each dimension comes
    from the distance in that dimension between the interpolation point and the
    "edge", and the total weight for the "edge" is the product of the weights
    for each dimension.
    The final interpolated value is the sum of contributions from each of the
    2**ndim "edges", where the contribution from each edge is the product of the
    edge value and its associated total weight.

    In practice the code is vectorised, so we do this for all interpolated
    points at once, and we use a slightly different order of calculations to
    minimise the work that needs to be done when repeating the interpolation on
    new data.
    """
    def __init__(self, in_points, out_shape):
        self.in_points = [np.asarray(p) for p in in_points]
        self.in_shape = tuple(len(p) for p in in_points)
        self.ndim = len(in_points)
        self.out_shape = tuple(out_shape)
        
        for p_arr in self.in_points:
            if p_arr.ndim != 1:
                raise ValueError("Points arrays must be 1D")
            if np.any(np.diff(p_arr) <= 0.):
                raise ValueError("Points arrays must be strictly ascending")
            if p_arr.size < 2:
                raise ValueError("Points arrays must have at least 2 values")
        if len(out_shape) != self.ndim:
            raise ValueError("The output array must have the same number of "
                             "dimensions as the input array")
        for n_p in out_shape:
            if n_p < 2:
                raise ValueError("Each output dimension needs at least 2 points")
        self.out_points = [np.linspace(p[0], p[-1], n_p) for p,n_p in zip(
                                                     self.in_points, out_shape)]
        
        # Find indices of the lower edge for each interpolated point in each
        # dimension:
        self.lower_edge_inds = []
        # We calculate the distance from the interpolated point to the lower
        # edge in units where the distance from the lower to the upper edge is 1.
        self.norm_distances = []
        # Iterate dimensions:
        for p_out, p_in in zip(self.out_points, self.in_points):
            # p_out and p_in are a series of coordinate values for this dimension
            i_vec = np.searchsorted(p_in, p_out) - 1
            np.clip(i_vec, 0, p_in.size - 2, out=i_vec)
            self.lower_edge_inds.append(i_vec)
            p_in_diff = np.diff(p_in) # p_in_diff[j] is p_in[j+1] - p_in[j]
            # Use fancy indexing:
            self.norm_distances.append((p_out - p_in[i_vec]) / p_in_diff[i_vec])

        # Find weights:
        self._find_weights()

        # Find fancy indices for each edge:
        prod_arr = cartesian_prod(self.lower_edge_inds)
        fancy_inds_lower = tuple(prod_arr[:,i] for i in range(self.ndim))
        # The fancy indices are for the edge which corresponds to the "lower"
        # edge position in each dimension, and will extract the edge values
        # from the input grid array for every interpolated point at once
        self.fancy_inds_all = {}
        for all_j in itertools.product(*[[0,1] for _ in range(self.ndim)]):
            self.fancy_inds_all[all_j] = tuple(a + j for j,a in zip(all_j,
                                                              fancy_inds_lower))
            # We do this calculation here and store the results because it is
            # surprsingly slow and otherwise we'd need to do it for every
            # emission line (this approach does take a lot of memory though)


    def _find_weights(self):
        """
        Find the weights that are necessary for linear interpolation
        """
        # Weights for upper edge for interpolation positions in each dimension:
        weights_upper = self.norm_distances
        # The norm_distances are from the lower edge.  The weighting is such
        # that if this distance is large, we favour the upper edge.
        upper_all = cartesian_prod(weights_upper)
        lower_all = 1 - upper_all
        # These two arrays have shape (n_iterp_points, ndim)
        weights_all_l_u = [lower_all, upper_all]

        # Calculate the weight for each edge.  We do this for every interpolated
        # point at once.
        # The 2**ndim edges are identified by keys (j0, j1, ..., jn) where
        # j == 0 is for the lower edge in a dimension; j == 1 is for the upper edge.
        weights = {} # We'll have a vector of weights for each edge; the vector
        # has one entry for each interpolated point
        for all_j in itertools.product(*[[0,1] for _ in range(self.ndim)]):
            combined_weights = np.ones(upper_all.shape[0]) # Length n_iterp_points
            for k,j in enumerate(all_j):
                combined_weights *= weights_all_l_u[j][:,k]
                # For j = 0 use "lower_all", and for j = 1, use "upper_all".
                # We multiply the weights for each dimension to obtain the total
                # weight for this edge for each interpolated point.
            weights[all_j] = combined_weights
            # The weights for this edge are in a 1D array, which has a length
            # equal to the total number of points in the grid.

        self.weights = weights


    def __call__(self, in_grid_values):
        """
        Evaluate linear interpolation to resample a regular grid.

        Parameters
        ----------
        in_grid_values : ndarray
            Array holding the grid values of the grid to be resampled.
        """
        if in_grid_values.shape != self.in_shape:
            raise ValueError("Shape of grid array doesn't match shape of this "
                             "RegularGridResampler")

        out_values = np.zeros(np.product(self.out_shape)) # 1D for now
        # Iterate edges, adding the contribution from each edge to the
        # interpolated values
        for all_j, edge_weights in self.weights.items():
            edge_fancy_inds = self.fancy_inds_all[all_j]
            out_values += in_grid_values[edge_fancy_inds] * edge_weights

        # Reshape the array
        out_grid_values = out_values.reshape(self.out_shape)

        return self.out_points, out_grid_values


