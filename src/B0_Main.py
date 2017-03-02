from __future__ import print_function, division
from collections import OrderedDict as OD
import numpy as np  # Core numerical library
import pandas as pd
# For interpolating an n-dimensional regular grid:
# import itertools # For finding Cartesian product and combinatorial combinations
from . import B1_Grid_working
# from . import B2_Prior
from . import B3_Posterior
from . import B4_Plotting


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

    def __init__(self, grid_file, grid_params, lines_list, **kwargs):
        """
        Initialise an instance of the Bigelm_model class.

        Required arguments to initialise the Bigelm_model instance:
        grid_file: The filename of an ASCII csv table containing photoionisation
                   model grid data in the form of a database table. Each
                   gridpoint (point in parameter space) is a row in this table.
                   The values of the grid parameters for each row are defined in
                   a column for each parameter.
                   No assumptions are made about the order of the gridpoints
                   (rows) in the table.  Spacing of grid values along an axis
                   may be uneven, but the full grid is required to a be a
                   regular, n-dimensional rectangular grid.  There is a column
                   of fluxes for each modelled emission line, and model fluxes
                   are assumed to be normalised to Hbeta
                   Any non-finite fluxes (e.g. nans) will be set to zero.
        grid_params: List of the unique names of the grid parameters as strings.
                     The order is the order of the grid dimensions, i.e. the
                     order in which arrays in bigelm will be indexed.
        lines_list: List of column names from grid_file corresponding to the
                    emission lines we'll be using in this Bigelm_model instance.

        Optional additional keyword arguments:
        interpd_grid_shape: A tuple of integers, giving the size of each
                            dimension of the interpolated flux grids.  The order
                            of the integers corresponds to the order of
                            parameters in grid_params.
                            The default is 15 gridpoints along each dimension.
                            Note that the values entered in interpd_grid_shape
                            have a major impact on the speed of the grid
                            interpolation.
        grid_error:         The systematic relative error on grid fluxes, as a
                            linear proportion.  Default is 0.35 (average of
                            errors of 0.15 dex above and below a value).
        """
        # Initialise and do some checks...
        print("Initialising BIGELM model...")

        # Determine if a custom interpolated grid shape was specified:
        interpd_grid_shape = kwargs.pop("interpd_grid_shape", [15]*len(grid_params))
        if len(interpd_grid_shape) != len(grid_params):
            raise ValueError("interpd_grid_shape should contain exactly "
                             "one integer for each parameter" )

        grid_rel_error = kwargs.pop("grid_error", 0.35)
        assert grid_rel_error > 0

        # Are there any remaining keyword arguments that weren't used?
        if len(kwargs) > 0:
            raise ValueError("Unknown keyword argument(s) " +
                                      ", ".join(str(k) for k in kwargs.keys()) )

        # Call grid initialisation:
        Raw_grids, Interpd_grids = B1_Grid_working.initialise_grids(grid_file,
                                    grid_params, lines_list, interpd_grid_shape)
        Raw_grids.grid_rel_error = grid_rel_error
        Interpd_grids.grid_rel_error = grid_rel_error
        self.Raw_grids = Raw_grids
        self.Interpd_grids = Interpd_grids



    def __call__(self, obs_fluxes, obs_flux_errors, obs_emission_lines, **kwargs):
        """
        Run BIGELM Bayesian parameter estimation using the grids saved in this
        Bigelm_model object.
        Required positional arguments:
        obs_fluxes:         list or array of observed emission-line fluxes
                            normalised to Hbeta
        obs_flux_errors:    list or array of corresponding measurement errors
        obs_emission_lines: list of corresponding emission line names as strings
        
        Optional keyword arguments which affect the parameter estimation:
        deredden:        De-redden observed fluxes to match the Balmer decrement
                         at each interpolated grid point?  Default True.
        obs_wavelengths: If deredden=True, you must also supply a list of
                         wavelengths (Angstroems) associated with obs_fluxes.
                         Default None.
        prior:           The prior to use when calculating the posterior.
                         Choices are: "Uniform", "SII_ratio", "He_ratio",
                         "SII_and_He_ratios", or a user-supplied function.
                         Default: "Uniform".
                         See the code file "B2_Prior.py" for the details
                         of the SII and He priors and to see the required inputs
                         and outputs for a user-defined function.
        
        Optional additional keyword arguments regarding outputs:
        param_display_names:  A dictionary of parameter display names for grid
                              parameters, for plotting purposes.  The dictionary
                              keys are parameter names in the grid file, and the
                              corresponding values are the "display" names.  The
                              display names can be raw strings (e.g. r"$\alpha$")
                              in order to include e.g. Greek letters.
                              Not all of the grid parameters need to be included 
                              in param_display_names; raw parameter names will
                              be used by default.
        posterior_plot:   A filename for saving out a results image of 2D and
                          1D marginalised posterior pdfs.  The figure will only
                          be generated and saved if this keyword is specified.
        prior_plot:       A filename; as for posterior_plot but for the prior
        likelihood_plot:  A filename; as for posterior_plot but for the likelihood
        estimate_table:   A filename for a csv file containing Bayesian
                          parameter estimates for the grid parameters.
        best_model_table: A filename for a csv file which will compare observed
                          and model fluxes at the point defined by the Bayesian
                          parameter estimates.
        table_on_plots:   Include a flux comparison table on the corner plots?
                          Default: True

        Returns a Bigelm_result object (defined in B3_Posterior.py), which
        contains all of the data relevant to the Bayesian parameter estimation
        as attributes.

        ################
        Other notes:
        We don't deal with measured emission-line fluxes that are provided summed for two lines.
        We ignore the possibility of including only upper bounds for emission-line fluxes
        We currently compare linear fluxes, not log fluxes...
        In finding marginalised posteriors, we use trapezium integration - perhaps dodgy, 
        but I think we have too few dimensions for Monte Carlo methods and I don't think it's
        worth doing higher-order numerical integration.
        At the moment zero model flux => zero systematic error on the flux. Wrong!
        ################
        """
        print("Running BIGELM...")

        Raw_grids = self.Raw_grids
        Interpd_grids = self.Interpd_grids

        deredden = kwargs.pop("deredden", True) # Default True
        assert isinstance(deredden, bool)
        obs_wavelengths = kwargs.pop("obs_wavelengths", None) # Default None
        if deredden and (obs_wavelengths is None):
            raise ValueError("Must supply obs_wavelengths if deredden==True")
        if (obs_wavelengths is not None) and not deredden:
            pass # obs_wavelengths is unnecessary but will be checked anyway.
        # Process the input observed data; DF_obs is a pandas DataFrame table
        # where the emission line names index the rows:
        DF_obs = process_observed_data(obs_fluxes, obs_flux_errors,
                                            obs_emission_lines, obs_wavelengths)

        input_prior = kwargs.pop("prior", "Uniform") # Default "Uniform"

        #----------------------------------------------------------------------
        # Handle options for BIGELM outputs:
        # Determine the list of parameter display names to use for plotting:
        param_display_names = Interpd_grids.param_names.copy() # Default
        if "param_display_names" in kwargs:
            custom_display_names = kwargs.pop("param_display_names")
            for i, custom_name in enumerate(custom_display_names):
                param_display_names[i] = custom_name # Override default
        # Include text "best model" table on posterior corner plots?
        table_on_plots = kwargs.pop("table_on_plots", True) # Default True
        # Filenames for output corner plot images?  Default None (no plotting)
        likelihood_plot = kwargs.pop("likelihood_plot", None)
        prior_plot      = kwargs.pop("prior_plot",      None)
        posterior_plot  = kwargs.pop("posterior_plot",  None)

        # Filenames for output csv tables?  Default None (don't write out table)
        estimate_table   = kwargs.pop("estimate_table",   None)
        best_model_table = kwargs.pop("best_model_table", None)

        # Are there any remaining keyword arguments that weren't used?
        if len(kwargs) > 0:
            raise ValueError("Unknown keyword argument(s): " +
                             ", ".join("'{0}'".format(k) for k in kwargs.keys()))

        #----------------------------------------------------------------------
        # Create a "Bigelm_result" object instance, which involves calculating
        # the prior, likelihood and posterior, along with parameter estimates:
        Result = B3_Posterior.Bigelm_result(Interpd_grids, DF_obs, deredden=deredden,
                                                      input_prior=input_prior)

        #----------------------------------------------------------------------
        # Outputs
        if estimate_table is not None: # Save parameter estimates table?
            Result.Posterior.DF_estimates.to_csv(estimate_table, index=True,
                                                            float_format='%.5f')
        if best_model_table is not None: # Save flux comparison table?
            Result.Posterior.DF_peak.to_csv(best_model_table, index=True,
                                                            float_format='%.5f')

        # Plot corner plots if requested:
        pdf_map = { "likelihood" : (likelihood_plot, Result.Likelihood),
                    "prior"      : (prior_plot,      Result.Prior     ),
                    "posterior"  : (posterior_plot,  Result.Posterior )  }
        for pdf_name, (out_image_name, Bigelm_nd_pdf) in pdf_map.items():
            if out_image_name == None:
                continue # Only do plotting if an image name was specified:
            if table_on_plots is True: # Include a fixed-width text table on image
                pd.set_option("display.precision", 4)
                plot_anno = "chi^2_r at {0} peak = {1:.2f}\n\n\n".format(
                                                   pdf_name, Bigelm_nd_pdf.chi2)
                plot_anno += ("Observed fluxes vs. model fluxes at the gridpoint of"
                              "\nparameter best estimates in the "+pdf_name+"\n")
                plot_anno += str(Bigelm_nd_pdf.DF_peak)
            else:
                plot_anno = None
            Bigelm_nd_pdf.Grid_spec.param_display_names = param_display_names
            print("Plotting corner plot for the", pdf_name, "...")
            B4_Plotting.plot_marginalised_ndpdf(out_image_name, Bigelm_nd_pdf,
                                                Raw_grids, plot_anno)

        
        print("Bigelm finished.")
        return Result




def process_observed_data(obs_fluxes, obs_flux_errors, obs_emission_lines,
                                                               obs_wavelengths):
    """
    Error check the input observed emission line data, and form it into a pandas
    DataFrame table.
    """
    obs_fluxes = np.asarray(obs_fluxes, dtype=float) # Ensure numpy array
    obs_flux_errors = np.asarray(obs_flux_errors, dtype=float)
    # Check measured data inputs all have the same length:
    n_measured = len(obs_emission_lines)
    if (obs_fluxes.size != n_measured) or (obs_flux_errors.size != n_measured):    
        raise ValueError("Input arrays obs_fluxes, obs_flux_errors and " 
                         "obs_emission_lines don't all have the same length.")
    if obs_wavelengths is not None:
        obs_wavelengths = np.asarray(obs_wavelengths, dtype=float)
        if obs_wavelengths.size != n_measured:
            raise ValueError("obs_wavelengths must have same length as obs_fluxes")
        # Some checks on the input wavelengths:
        if np.sum( ~np.isfinite(obs_wavelengths) ) > 0: # Any non-finite?
            raise ValueError("The wavelength for an emission line isn't finite.")
        if np.sum( obs_wavelengths <= 0 ) != 0: # All positive?
            raise ValueError("The wavelength for an emission line not positive.")

    # Some checks on the input measured fluxes:
    if np.sum( ~np.isfinite(obs_fluxes) ) > 0: # Any non-finite?
        raise ValueError("The measured flux for an emission line isn't finite.")
    if np.sum( obs_fluxes < 0 ) != 0: # Are all flux values are non-negative?
        raise ValueError("The measured flux for an emission line is negative.")
    
    # Some checks on the input measured flux errors:
    if np.sum( ~np.isfinite(obs_flux_errors) ) > 0: # Any non-finite?
        raise ValueError("The flux error for an emission line isn't finite.")
    if np.sum( obs_flux_errors <= 0 ) != 0: # All positive?
        raise ValueError("The flux error for an emission line is not positive.")

    # Form the data from the observations into a pandas DataFrame table.
    obs_dict = OD([("Line",obs_emission_lines)])
    if obs_wavelengths is not None:
        obs_dict["Wavelength"] = obs_wavelengths
    obs_dict["Flux"] = obs_fluxes
    obs_dict["Flux_err"] = obs_flux_errors
    DF_obs = pd.DataFrame(obs_dict)
    DF_obs.set_index("Line", inplace=True) # Row index is the emission line name

    return DF_obs


