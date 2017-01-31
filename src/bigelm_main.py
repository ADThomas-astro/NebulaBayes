from __future__ import print_function, division
import numpy as np  # Core numerical library
# For interpolating an n-dimensional regular grid:
# import itertools # For finding Cartesian product and combinatorial combinations
# from . import bigelm_plotting
from . import bigelm_grid_interpolation
from . import bigelm_posterior
from .bigelm_classes import Bigelm_container
# from .corner import corner  # For MCMC
from . import bigelm_plotting
from collections import OrderedDict as OD


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
    obs_fluxes:         list or array of observed emission-line fluxes
                        normalised to Hbeta
    obs_flux_errors:    list or array of corresponding measurement errors
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
    table_out:            A filename for a csv file containing Bayesian parameter
                          estimates for the grid.
    output_grids:         A Boolean, False by default.  If True, the outputted Results object will
                          contain the raw grids ("Raw_grids") object as an
                          as attribute.
                          The Raw_grids object is an instance of the Bigelm_grid class defined in this module.
    # interpd_grid_shape:   A tuple of integers, giving the size of each dimension of the interpolated
    #                       grid.  The order of the integers corresponds to the order of parameters in grid_params.
    #                       The default is 30 gridpoints along each dimension.  Note that the number of
    #                       interpolated gridpoints entered in interpd_grid_shape
    #                       may have a major impact on the speed of the program.
    #                       This keyword may only be supplied if the "Grid_container" keyword is not used.
    #                       Will be passed to function initialise_grids.
    n_interp_pts:         Number of interpolated points for 1D and 2D marginalised
                          posteriors and parameter estimation; default 100
    param_display_names:  A dictionary of display names for grid parameters, for plotting purposes.
                          A dictionary key is the parameter name in the grid file, and the corresponding
                          value its display name.
                          Can be raw strings (i.e. r"string") in order to include e.g. Greek letters.
                          Not all of the grid parameters need to be included in param_display_names;
                          raw parameter names will be used as display names by default.
                          This keyword may only be supplied if the "Grid_container" keyword is not used
                          (the previous param_display_names will be used).
    nburn:                Default 100
    n_iter:               Default 100
    burnchainplot         Filename for plot.  Default None
    chainplot             Filename for plot.  Default None
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
    if (n_measured != len(obs_fluxes) or n_measured != len(obs_flux_errors)):    
        raise ValueError("Input arrays obs_fluxes, obs_flux_errors " + 
                         "and obs_emission_lines don't have the same length.")
    obs_fluxes = np.array(obs_fluxes) # Ensure numpy array
    obs_flux_errors = np.array(obs_flux_errors)

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
        Grid_container = Bigelm_container()
        Grid_container.Params    = In_results.Params
        Grid_container.Raw_grids = In_results.Raw_grids
        Grid_container.flux_interpolators = In_results.flux_interpolators
        Params = Grid_container.Params
        Raw_grids = Grid_container.Raw_grids
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
        Grid_container = bigelm_grid_interpolation.initialise_grids(grid_file,
                                                  grid_params, lines_list)
        # Grid_container has attributes Params and Raw_grids
        Params = Grid_container.Params
        Raw_grids = Grid_container.Raw_grids

    #------------------------------------------------------------------------
    # Now the Grid_container and Params and Raw_grids are defined.

    # Determine the dictionary of parameter display names to use for plotting:
    Params.display_names = OD([(p,p) for p in Params.names])
    if "param_display_names" in kwargs:
        custom_display_names = kwargs.pop("param_display_names")
        for x in custom_display_names:
            Params.display_names[x] = custom_display_names[x] # Override default

    image_out = None # Default - no plotting if image_out==None
    if "image_out" in kwargs: # If it was specified, we'll save a plot.
        image_out = kwargs.pop("image_out")

    table_out = None # Default - don't save table if table_out==None
    if "table_out" in kwargs: # If it was specified, we'll save a plot.
        table_out = kwargs.pop("table_out")

    # Determine if we should include the model flux grids in the output:
    output_grids = False # Default - don't include raw and interpolated grids in output
    if "output_grids" in kwargs: # If output_grids was specified
        output_grids = kwargs.pop("output_grids")

    # nburn, niter = 100, 100 # Defaults
    # if "nburn" in kwargs:
    #     nburn = kwargs.pop("nburn")
    # if "niter" in kwargs:
    #     niter = kwargs.pop("niter")

    # burnchainplot, chainplot = None, None # Defaults
    # if "burnchainplot" in kwargs:
    #     burnchainplot = kwargs.pop("burnchainplot")
    # if "chainplot" in kwargs:
    #     chainplot = kwargs.pop("chainplot")

    n_interp_pts = 100 # Default
    if "n_interp_pts" in kwargs: # If n_interp_pts was specified
        n_interp_pts = kwargs.pop("n_interp_pts")

    # Ensure there aren't any remaining keyword arguments that we haven't used:
    if len(kwargs) != 0:
        raise ValueError( "Unknown or unnecessary keyword argument(s) " +
                          str(kwargs.keys())[1:-1] )        

    # #--------------------------------------------------------------------------
    Obs_Container = Bigelm_container()
    Obs_Container.lines_list = lines_list
    Obs_Container.obs_fluxes = obs_fluxes
    Obs_Container.obs_flux_errors = obs_flux_errors
    # # Run MCMC sampling
    # # sampler = bigelm_mcmc.fit_MCMC(Grid_container, Obs_Container, nwalkers=40,
    # #             nburn=nburn, niter=niter, burnchainplot=burnchainplot, chainplot=chainplot)
    # # print("Mean acceptance fraction: {0:.3f}"
    # #             .format(np.mean(sampler.acceptance_fraction)))

    #--------------------------------------------------------------------------
    # Calculate N-dimensional posterior array
    posterior = bigelm_posterior.calculate_posterior(Grid_container,
                    Obs_Container, log_prior_func=bigelm_posterior.uniform_prior)

    #--------------------------------------------------------------------------
    # Marginalise (and normalise) posterior
    marginalised_posteriors_1D, marginalised_posteriors_2D, posterior = \
                       bigelm_posterior.marginalise_posterior(posterior, Params,
                                                             Raw_grids.val_arrs)

    # Interpolate 1D and 2D posteriors
    posteriors_1D_interp, posteriors_2D_interp, param_val_arrs_interp = \
            bigelm_grid_interpolation.interpolate_posteriors(Raw_grids, Params,
           marginalised_posteriors_1D, marginalised_posteriors_2D, n_interp_pts)

    #--------------------------------------------------------------------------
    # Do Bayesian parameter estimation
    DF_estimates = bigelm_posterior.make_parameter_estimate_table(
                                    posteriors_1D_interp, param_val_arrs_interp)
    if table_out is not None:  # Save out if requested
        DF_estimates.to_csv(table_out, index=False, float_format='%.5f')

    #--------------------------------------------------------------------------
    # Plot a corner plot if requested
    if image_out != None: # Only do plotting if an image name was specified:
        bigelm_plotting.plot_marginalised_posterior(image_out, Params,
                                Raw_grids, param_val_arrs_interp, 
                                    posteriors_1D_interp, posteriors_2D_interp)
        
        # # MCMC corner plot
        # samples = sampler.flatchain.reshape((-1, Raw_grids.ndim))
        # # Each row of "samples" is a sample in the parameter space
        # display_labels = list(Params.display_names.values())
        # fig = corner(samples, labels=display_labels, range=[0.5]*5)
        #              # range=Raw_grids.p_minmax)#,
        #               # truths=[m_true, b_true, np.log(f_true)])
        # fig.savefig(image_out)
    

    #--------------------------------------------------------------------------
    Results = Bigelm_container()
    Results.Params = Params
    Results.DF_estimates = DF_estimates
    # Results.posterior = posterior
    if output_grids:
        Results.Raw_grids = Raw_grids
    print("Bigelm finished.")
    return Results




