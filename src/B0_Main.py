from __future__ import print_function, division
# from collections import OrderedDict as OD
import numpy as np  # Core numerical library
import pandas as pd
# For interpolating an n-dimensional regular grid:
# import itertools # For finding Cartesian product and combinatorial combinations
from . import B1_Grid_working
from . import B2_Bayes
from . import B3_Plotting


"""
Adam D. Thomas 2015 - 2017


BIGELM: Bayesian comparison of photoIonisation model Grids to Emission Line
        Measurements
        (Compulsory contrived acronym, as is the norm in modern astronomy)

The Bigelm_model class is designed for performing Bayesian parameter estimation.
The observed data are a set of emission line flux measurements with associated
errors.  The model data is a photoionisation model, varied over n=2 or more
parameters, input as
an n-dimensional grid of predicted emission line fluxes.  The measured
and modelled emission line fluxes are compared to calculate a "likelihood"
probability distribution, before Bayes' Theroem is applied to produce an
n-dimensional "posterior" probability distribution for the values of the
parameters.

Bigelm is heavily based on IZI (Blanc+2015).

"""



class Bigelm_model(object):
    """
    Primary class for working with Bigelm.  To use, initialise a class instance
    with a grid and then call the instance to run Bayesian parameter estimation.
    """

    def __init__(self, grid_file, grid_params, lines_list, interpd_grid_shape=None):
        """
        Initialise an instance of the Bigelm_model class.

        Required arguments to initialise the Bigelm_model instance:
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
        lines_list:     List of column names from grid_file corresponding to the
                        emission lines we'll be using in this Bigelm_model instance.

        Optional additional keyword arguments:
        interpd_grid_shape:   A tuple of integers, giving the size of each dimension of the interpolated
                              grid.  The order of the integers corresponds to the order of parameters in grid_params.
                              The default is 15 gridpoints along each dimension.  Note that the number of
                              interpolated gridpoints entered in interpd_grid_shape
                              may have a major impact on the speed of the program.
                              This keyword may only be supplied if the "Grid_container" keyword is not used.
                              Will be passed to function initialise_grids.
        """
        # Initialise and do some checks...
        print("Initialising BIGELM model...")

        # Determine if a custom interpolated grid shape was specified:
        if interpd_grid_shape is not None:
            if len(interpd_grid_shape) != len(grid_params):
                raise ValueError("interpd_grid_shape should contain exactly "
                                 "one integer for each parameter" )
        else:
            interpd_grid_shape = [15]*len(grid_params) # Default 

        # Call grid initialisation:
        Raw_grids, Interpd_grids = B1_Grid_working.initialise_grids(grid_file,
                                    grid_params, lines_list, interpd_grid_shape)
        self.Raw_grids = Raw_grids
        self.Interpd_grids = Interpd_grids



    def __call__(self, obs_fluxes, obs_flux_errors, obs_emission_lines, **kwargs):
        """
        Run BIGELM Bayesian parameter estimation using the grids saved in this Bigelm_model object.
        Required arguments:
        obs_fluxes:         list or array of observed emission-line fluxes
                            normalised to Hbeta
        obs_flux_errors:    list or array of corresponding measurement errors
        obs_emission_lines: a list of corresponding emission line names as strings
        
        Optional additional keyword arguments:
        deredden:             Add a parameter to the grid that is the extinction of
                              the observed fluxes.  The model grid fluxes will
                              be compared with reddening-corrected observed fluxes.  Default False.
        obs_wavelengths:      If deredden=True, you must also supply a list of wavelengths (Angstroems)
                              associated with obs_fluxes.  Default None.
        image_out:            A filename for saving out a results image of 2D and 1D marginalised posterior pdfs.
                              The figure will only be generated and saved if this keyword parameter is specified.
        table_out:            A filename for a csv file containing Bayesian parameter
                              estimates for the grid.
        param_display_names:  A dictionary of display names for grid parameters, for plotting purposes.
                              A dictionary key is the parameter name in the grid file, and the corresponding
                              value its display name.
                              Can be raw strings (i.e. r"string") in order to include e.g. Greek letters.
                              Not all of the grid parameters need to be included in param_display_names;
                              raw parameter names will be used as display names by default.
        #priors???            NOT IMPLETMENTED: A dictionary of functions for priors for each parameter... ???


        Returns a Bigelm_container object (defined in this module), which contains
        the following attributes...

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
        print("Running BIGELM...")
        # Params = self.Grid_container.Params
        # Interpd_grids = self.Grid_container.Interpd_grids
        Raw_grids = self.Raw_grids
        Interpd_grids = self.Interpd_grids


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

        # Determine the list of parameter display names to use for plotting:
        param_display_names = Interpd_grids.param_names.copy() # Default
        if "param_display_names" in kwargs:
            custom_display_names = kwargs.pop("param_display_names")
            for i, custom_name in enumerate(custom_display_names):
                param_display_names[i] = custom_name # Override default

        # Deredden observed fluxes and have extra grid parameter for extinction?
        deredden = kwargs.pop("deredden", False)
        # Observed wavelengths?
        obs_wavelengths = kwargs.pop("obs_wavelengths", None)
        if deredden and (obs_wavelengths is None):
            raise ValueError("Must supply obs_wavelengths if deredden=True")
        if obs_wavelengths is not None:
            if not deredden:
                pass # obs_wavelengths is unnecessary; don't check or use it
            elif len(obs_wavelengths) != n_measured:
                raise ValueError("obs_wavelengths must have same length as obs_fluxes")
        # Output corner plot image: Default is no plotting (None)
        image_out = kwargs.pop("image_out", None)
        # Output parameter estimate table: Default is no plotting (None)
        table_out = kwargs.pop("table_out", None)

        # Ensure there aren't any remaining keyword arguments that we haven't used:
        if len(kwargs) > 0:
            raise ValueError( "Unknown or unnecessary keyword argument(s) " +
                              str(kwargs.keys())[1:-1] )

        #----------------------------------------------------------------------
        class dummy(object):
            pass
        Obs_Container = dummy()
        Obs_Container.lines_list = lines_list
        Obs_Container.obs_fluxes = obs_fluxes
        Obs_Container.obs_flux_errors = obs_flux_errors
        Obs_Container.obs_wavelengths = obs_wavelengths

        #----------------------------------------------------------------------
        # Create a "Bigelm_result" object instance, which involves calculating
        # the posterior and parameter estimates:
        Result = B2_Bayes.Bigelm_result(Interpd_grids, Obs_Container,
                                              deredden=deredden,
                                          log_prior_func=B2_Bayes.uniform_prior)
        Result.Grid_spec.param_display_names = param_display_names

        # Save out results table if requested
        if table_out is not None:
            Result.DF_estimates.to_csv(table_out, index=False, float_format='%.5f')

        # Plot a corner plot if requested
        if image_out != None: # Only do plotting if an image name was specified:
            plot_text = "chi^2_r at posterior peak = {0:.2f}\n\n\n".format(Result.chi2)
            plot_text += "Observed fluxes vs. model fluxes at posterior peak\n"
            pd.set_option("display.precision", 4)
            plot_text += str(Result.DF_peak) # To print in a monospace font
            B3_Plotting.plot_marginalised_posteriors(image_out, Result,
                                                           Raw_grids, plot_text)

        print("Bigelm finished.")
        return Result


