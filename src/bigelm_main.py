from __future__ import print_function, division
import numpy as np  # Core numerical library
# For interpolating an n-dimensional regular grid:
import itertools # For finding Cartesian product and combinatorial combinations
from . import bigelm_plotting
from . import bigelm_grid_interpolation
from .bigelm_classes import Bigelm_container


"""
Adam D. Thomas 2015 - 2016

"""



#============================================================================
def run_bigelm(obs_fluxes, obs_flux_errors, obs_emission_lines, **kwargs):
    """
    BIGELM: Bayesian comparison of photoIonisation model Grids to Emission Line
            Measurements
            (Compulsory contrived acronym, as is the norm in modern astronomy)

    This function performs Bayesian parameter estimation.  The data are a set
    of emission line flux measurements with associated errors.  The model
    is a photoionisation model, varied over n=2 or more parameters, input as
    an n-dimensional grid of predicted emission line fluxes.  The measured
    and modelled emission line fluxes are compared to calculate a "likelihood"
    probability distribution, before Bayes' Theroem is applied to produce an
    n-dimensional "posterior" probability distribution for the values of the
    parameters.

    Bigelm is heavily based on IZI (Blanc+2015).

    Returns a Bigelm_container object (defined in this module), which contains
    the following attributes...

    Optionally the returned class instance may also have attributes containing
    the model grid information (see output_grids keyword below).

    Required arguments to bigelm:
    obs_fluxes:         a numpy array of observed emission-line fluxes
                        normalised to Hbeta
    obs_flux_errors:    a numpy array of corresponding measurement errors
    obs_emission_lines: a list of corresponding emission line names as strings
    and EITHER the following two keyword arguments:
        grid_file:      the filename of an ASCII csv table containing photoionisation model
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
        grid_params:    List of the unique names of the grid parameters as strings.
                        The order is the order of the grid dimensions, i.e. the order
                        in which arrays in bigelm will be indexed.
    OR the following single keyword argument:
        Grid_container: An instance of the Bigelm_container class defined in
                        this module, containing grid data.  This will either be
                        the output of a previous run of bigelm (which had
                        output_grids=True; see below), or a pre-prepared output
                        of the initialise_grids function in this module.
                        Using this keyword will ensure bigelm uses the
                        previously calculated "Params", "Raw_grids" and
                        "Interpd_grids" attributes of the Bigelm_container class
                        instance.  Avoiding recalculation of the raw and interpolated
                        flux grids for each emission line saves a very large fraction of the bigelm
                        computation time.  Note that the "Params" attribute is an instance of the
                        Grid_parameters class defined in this module; the "Raw_grids" and "Interpd_grids"
                        attributes are instances of the Bigelm_grid class defined in this module.  Any 
                        contents of Grid_container other than these three attributes are ignored.

    Optional additional keyword arguments:
    image_out:            A filename for saving out a results image of 2D and 1D marginalised posterior pdfs.
                          The figure will only be generated and saved if this keyword parameter is specified.
    output_grids:         A Boolean, False by default.  If True, the outputted Results object will
                          contain the raw grids ("Raw_grids") and interpolated grids ("Interpd_grids") objects
                          as attributes.  Note that the interpolated grids object may be large (e.g. 6 Mb * 50 lines)
                          The Raw_grids and Interpd_grids objects are instances of the Bigelm_grid class defined in this module.
    # interpd_grid_shape:   A tuple of integers, giving the size of each dimension of the interpolated
    #                       grid.  The order of the integers corresponds to the order of parameters in grid_params.
    #                       The default is 30 gridpoints along each dimension.  Note that the number of
    #                       interpolated gridpoints entered in interpd_grid_shape
    #                       may have a major impact on the speed of the program.
    #                       This keyword may only be supplied if the "Grid_container" keyword is not used.
    #                       Will be passed to function initialise_grids.
    param_display_names:  A dictionary of display names for grid parameters, for plotting purposes.
                          A dictionary key is the parameter name in the grid file, and the corresponding
                          value its display name.
                          Can be raw strings (i.e. r"string") in order to include e.g. Greek letters.
                          Not all of the grid parameters need to be included in param_display_names;
                          raw parameter names will be used as display names by default.
                          This keyword may only be supplied if the "Grid_container" keyword is not used
                          (the previous param_display_names will be used).
    #priors???            NOT IMPLETMENTED: A dictionary of functions for priors for each parameter... ???
    #extra_lines:         NOT IMPLEMENTED: A list of additional emission lines to make grids for.  Useful for
                          making grids for use in later bigelm runs.  A union of obs_emission_lines and extra_lines
                          is used to make the final list of emission lines for grids.


    Other notes:
    We don't deal with measured emission-line fluxes that are provided summed for two lines.
    We ignore the possibility of including only upper bounds for emission-line fluxes
    We currently assume uniform priors for everything
    We currently compare linear fluxes, not log fluxes...
    In calculating the likelihood we assume a systematic error on the normalised model fluxes of 0.15dex.
    In finding marginalised posteriors, we use trapezium integration - perhaps dodgy, 
    but I think we have too few dimensions for Monte Carlo methods and I don't think it's
    worth doing higher-order numerical integration.
    At the moment zero model flux => zero systematic error on the flux. Wrong!

    """
    # Initialise and do some checks...
    print("Initialising bigelm...")

    # Check types??

    # Check measure data inputs all have the same length:
    n_measured = len(obs_emission_lines)
    if (n_measured != obs_fluxes.size or n_measured != obs_flux_errors.size):    
        raise ValueError("Input arrays obs_fluxes, obs_flux_errors " + 
                         "and obs_emission_lines don't have the same length.")

    # We'll consider only the emission lines in the input measured fluxes,
    # and ignore any other emission lines provided in the model grid:
    lines_list = obs_emission_lines # Emission lines to work with

    # Some checks on the input measured fluxes:
    # Check that all flux values are finite:
    if np.sum( np.logical_not( np.isfinite( obs_fluxes ) ) ) != 0:
        raise ValueError("The measured flux for an emission line isn't finite.")
    # Check that all flux values are positive:
    if np.sum( obs_fluxes < 0 ) != 0:
        raise ValueError("The measured flux for an emission line is negative.")
    
    # Some checks on the input measured flux errors:
    # Check that all errors are finite:
    if np.sum( np.logical_not( np.isfinite( obs_flux_errors ) ) ) != 0:
        raise ValueError("The flux error for an emission line isn't finite.")
    # Check that all errors are positive:
    if np.sum( obs_flux_errors < 0 ) != 0:
        raise ValueError("The flux error for an emission line is negative.")


    # In processing the keyword arguments in the dictionary "kwargs",
    # we use the "kwargs.pop(key1)" method, which removes the item (key1, val1)
    # from the dictionary, and returns the associated quantity val1.
    if "Grid_container" in kwargs:
        # We use the Bigelm_container output object that was supplied:
        In_results = kwargs.pop("Grid_container")
        # Unpack the Grid_parameters and Bigelm_grid objects:
        Params        = In_results.Params
        Raw_grids     = In_results.Raw_grids
        Interpd_grids = In_results.Interpd_grids
        # We ignore the other attributes of the In_results object.

        # Check that all emission lines in input are also in grids:
        # We only need to check one of Raw_grids and Interpd_grids
        for line in lines_list:
            if not line in Raw_grids.grids:
                raise ValueError("No model grid was found for measured" +
                                 " emission line " + line)
    else:
        # We need to construct the Grid_parameters and Bigelm_grid objects
        # using the input grid_file and grid_params keyword arguments
        if "grid_file" in kwargs:
            grid_file = kwargs.pop("grid_file")
        else:
            raise ValueError("grid_file must be provided if "   + 
                             "Grid_container is not provided"   )

        if "grid_params" in kwargs:
            grid_params = kwargs.pop("grid_params")
        else:
            raise ValueError("grid_params must be provided if " + 
                             "Grid_container is not provided"   )

        # interpd_grid_shape = None # Default to pass into XXX function
        # # Determine if a custom interpolated grid shape was specified:
        # if "interpd_grid_shape" in kwargs:
        #     interpd_grid_shape = kwargs.pop("interpd_grid_shape")


        # Call grid initialisation:
        Grid_init = bigelm_grid_interpolation.initialise_grids(grid_file,
                                                  grid_params, lines_list)
        # Unpack:
        Params        = Grid_init.Params
        Raw_grids     = Grid_init.Raw_grids
        # Interpd_grids = Grid_init.Interpd_grids

    # The following keyword arguments are relevant whether or
    # not "Grid_container" was specified:
    # Determine if an output image file name was specified:

    # Determine the dictionary of parameter display names to use for plotting:
    Params.display_names = dict([(p, p) for p in Params.names])  # Default
    if "param_display_names" in kwargs:
        custom_display_names = kwargs.pop("param_display_names")
        for x in custom_display_names:
            Params.display_names[x] = custom_display_names[x] # Override default

    image_out = None # Default - no plotting if image_out==None
    if "image_out" in kwargs: # If it was specified, we'll save a plot.
        image_out = kwargs.pop("image_out")

    # Determine if we should include the model flux grids in the output:
    output_grids = False # Default - don't include raw and interpolated grids in output
    if "output_grids" in kwargs: # If output_grids was specified
        output_grids = kwargs.pop("output_grids")

    # Ensure there aren't any remaining keyword arguments that we haven't used:
    if len(kwargs) != 0:
        raise ValueError( "Unknown or unnecessary keyword argument(s) " +
                          str(kwargs.keys())[1:-1] )        



    #--------------------------------------------------------------------------
    # Calculate the likelihood, prior and posterior...
    
    print("Calculating posterior over full grid...")
    # The posterior will have the same shape as an interpolated model grid
    # for one of the emission lines

    # Use systematic uncertainty in modelled fluxes, as in Blanc et al.
    epsilon  = 0.15 # dex.  Default is 0.15 dex systematic uncertainty
    # Convert from dex to a scaling factor:
    epsilon_2 = 10**epsilon - 1  # This gives a factor of 0.41 for epsilon=0.15 dex
    # epsilon_2 is a scaling factor to go from a linear modelled flux to an
    # associated systematic error
    # Note the original from izi.pro is equivalent to:
    # epsilon_2 = epsilon * np.log(10)
    # This gives 0.345 for epsilon=0.15 dex. I don't understand this.
    # Why was a log used?  And a log base e?  This is surely wrong.
    # What is intended by this formula anyway?
    # Note that epsilon_2 is the assumed factional systematic error in the model
    # fluxes already normalised to Hbeta.  In izi.pro the default is 0.15 dex,
    # but the default is given as 0.1 dex in the Blanc+15 paper.
    
    # Calculate uniform prior:
    prior = ( np.ones( Interpd_grids.shape, dtype="double" ) / 
              np.product( Interpd_grids.shape )                 )

    # Initialise ln(likelihood) with ones:
    log_likelihood = np.zeros( Interpd_grids.shape, dtype="double")

    # Calculate likelihood (everything has been leading to this moment!)
    for i, emission_line in enumerate(lines_list):
        # Use a log version of equation 3 on pg 4 of Blanc et al. 2015
        # N.B. var has shape Interpd_grids.shape, and is the sum of variances due to
        # both the measured and modelled fluxes
        var = obs_flux_errors[i]**2 + (epsilon_2 * Interpd_grids.grids[emission_line])**2
        log_likelihood += ( - (( obs_fluxes[i] - Interpd_grids.grids[emission_line] )**2 / 
                                 ( 2.0 * var ))  -  0.5*np.log( var ) )  # N.B. "log" is base e
    
    # Perform a pseudo-normalisation to prevent numerical overflow, then find linear likelihood:
    log_likelihood -= np.max(log_likelihood)
    likelihood = np.exp(log_likelihood)

    # Calculate (un-normalised) posterior as the product of the likelihood and the prior:
    # Bayes' Theorem:
    posterior = likelihood * prior  # Has shape Interpd_grids.shape
    # Note:  The parameter space, space of measurements and space of predictions are all continuous.
    # Each value P in the posterior array is differential, i.e. P*dx = (posterior)*dx
    # for a vector of parameters x.

    #--------------------------------------------------------------------------
    # Marginalise posteriors:
    posterior, marginalised_posteriors_2D, marginalised_posteriors_1D = \
            marginalise_posteriors(posterior, Params)
    
    #--------------------------------------------------------------------------
    # Plot the 2D and 1D marginalised posterior pdfs if requested
    if image_out != None: # Only do plotting if an image name was specified:
        bigelm_plotting.plot_marginalised_posterior(image_out, Params,
                                Raw_grids, Interpd_grids, 
                        marginalised_posteriors_1D, marginalised_posteriors_2D)

    #--------------------------------------------------------------------------
    # Combine the results into a Bigelm_container object, and return it...
    # Use capitalised names to denote instances of custom classes

    Results = Bigelm_container() # Initialise class instance
    Results.Params = Params
    Results.prior = prior
    Results.posterior = posterior # Not used as is here, but user might want it
    Results.marginalised_posteriors_1D = marginalised_posteriors_1D
    Results.marginalised_posteriors_2D = marginalised_posteriors_2D
    # Include raw and interpolated model grids in output if requested by user:
    if output_grids:
        Results.Raw_grids = Raw_grids
        Results.Interpd_grids = Interpd_grids

    print("Bigelm finished.")

    return Results



#============================================================================
def marginalise_posteriors(posterior, Params, Interpd_grids):
    """
    Marginalise a 3D posterior array.
    posterior: The full 3D posterior
    Params: Grid_parameters instance
    Interpd_grids: Bigelm_grid instance holding interpolated model grids.
    
    Returns the 3D posterior, a dictionary of 2D posteriors, and
    a dictionary of 1D posteriors, all normalised.
    """
    #--------------------------------------------------------------------------
    # Calculate the 2D marginalised posterior pdf for every possible combination
    # of 2 parameters

    print("Calculating 2D marginalised posteriors...")
    # List of all possible pairs of two parameter names:
    Params.double_names = list( itertools.combinations( Params.names, 2) )
    # Looks something like: [('BBBP','Gamma'), ('BBBP','NT_frac'),
    #                        ('BBBP','UH_at_r_inner'), ('Gamma','NT_frac'),
    #                        ('Gamma','UH_at_r_inner'), ('NT_frac','UH_at_r_inner')]
    # Corresponding list of possible combinations of two parameter indices:
    Params.double_indices = list( itertools.combinations( np.arange(Params.n_params), 2) )
    # Looks something like: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    # Note that the order of indices in each tuple is from samller to larger
    
    # Initialise dictionary of all possible 2D marginalised posterior arrays:
    marginalised_posteriors_2D = {}  # The dict keys will be tuples of 2 parameter names.
    # Iterate over all possible pairs of parameters:
    for double_name, param_inds_double in zip(Params.double_names, Params.double_indices):
        # Generate list of indices/dimensions/parameters to integrate over:
        inds_for_integration = np.arange( Params.n_params ).tolist()  # Initialise
        inds_for_integration.remove( param_inds_double[0] )
        inds_for_integration.remove( param_inds_double[1] )
        inds_for_integration.reverse() # Ensure we integrate over higher dimensions first,
        # so dimension index numbers are still valid after each integration.

        marginalised_posteriors_2D[double_name] = posterior.copy()  # Initialise
        # Keep integrating one dimension at a time until the result only has 2 dimensions:
        for param_index in inds_for_integration:
            # Integrate over this dimension (parameter), using the trapezoidal rule
            marginalised_posteriors_2D[double_name] = np.trapz( 
                marginalised_posteriors_2D[double_name], axis=param_index,
                dx=Interpd_grids.spacing[param_index] )

    #--------------------------------------------------------------------------
    # Calculate the 1D marginalised posterior pdf for each individual parameter

    print("Calculating 1D marginalised posteriors...")
    # Initialise dictionary of all 1D marginalised posterior arrays:
    marginalised_posteriors_1D = {}

    # For the first parameter in Params.names:
    # Integrate the first 2D marginalised posterior pdf over the other
    # dimension (parameter), using the trapezoidal rule:
    marginalised_posteriors_1D[Params.names[0]] = np.trapz( 
            marginalised_posteriors_2D[Params.double_names[0]], axis=1,
            dx=Interpd_grids.spacing[1] )

    # For all parameters after the first in Params.names:
    for double_name, param_inds_double in zip(Params.double_names[:Params.n_params-1],
                                              Params.double_indices[:Params.n_params-1]):
        # For each pair of parameters we take the second parameter, and integrate 
        # over the first parameter of the pair (which by construction is always the
        # first parameter in Params.names).
        assert( param_inds_double[0] == 0 )
        param = Params.names[ param_inds_double[1] ]
        # Integrate over first dimension (parameter) using trapezoidal method:
        marginalised_posteriors_1D[param] = np.trapz( marginalised_posteriors_2D[double_name],
                                                     axis=0, dx=Interpd_grids.spacing[0] )

    #--------------------------------------------------------------------------
    # Calculate the 0D marginalised posterior pdf (by which I mean find
    # the normalisation constant - the 0D marginalised posterior should be 1!)
    # Then normalise the 1D and 2D marginalised posteriors:

    # Firstly find the integral over all Params.n_params dimensions by picking
    # any 1D marginalised posterior (we use the first) and integrating over it:
    integral = np.trapz( marginalised_posteriors_1D[ Params.names[0] ],
                         dx=Interpd_grids.spacing[0] )
    print( "Integral for un-normalised full posterior is " + str(integral) )
    # Now actually normalise each 2D and 1D marginalised posterior:
    for double_name in Params.double_names:
        # Divide arrays in-place in memory:
        marginalised_posteriors_2D[double_name] /= integral
    for param in Params.names:
        # Divide arrays in-place in memory:
        marginalised_posteriors_1D[param] /= integral
    # Now normalise the full posterior, since we output it and the user
    # might want it normalised:
    posterior /= integral

    return posterior, marginalised_posteriors_2D, marginalised_posteriors_1D



