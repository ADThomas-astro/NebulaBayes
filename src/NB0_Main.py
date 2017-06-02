from __future__ import print_function, division
from collections import OrderedDict as OD
import numpy as np  # Core numerical library
import pandas as pd # For tables ("DataFrame"s)
from . import NB1_Grid_working
from . import NB3_Bayes
from .NB4_Plotting import ND_PDF_Plotter


"""
NebulaBayes
Adam D. Thomas
Research School of Astronomy and Astrophysics
Australian National University
2015 - 2017

The NB_Model class in this module is the entry point for performing Bayesian
parameter estimation.  The data are a
set of emission line flux measurements with associated errors.  The model
is a photoionisation model, varied in a grid over n=2 or more parameters,
input as n-dimensional grids of fluxes for each emission line.  The model is
for an HII region or AGN Narrow Line Region, for example.  The measured
and modelled emission line fluxes are compared to calculate a "likelihood"
probability distribution, before Bayes' Theorem is applied to produce an
n-dimensional "posterior" probability distribution for the values of the
parameters.  The parameter values are estimated from 1D marginalised
posteriors.

NebulaBayes is heavily based on IZI (Blanc+2015).

"""



class NB_Model(object):
    """
    Primary class for working with NebulaBayes.  To use, initialise a class instance
    with a grid and then call the instance to run Bayesian parameter estimation.
    """

    def __init__(self, grid_file, grid_params, lines_list, **kwargs):
        """
        Initialise an instance of the NB_Model class.

        Required arguments to initialise the NB_Model instance:
        grid_file: The filename of an ASCII csv table containing photoionisation
                   model grid fluxes in the form of a database table. Each
                   gridpoint (point in parameter space) is a row in this table.
                   The values of the grid parameters for each row are defined in
                   a column for each parameter.
                   No assumptions are made about the order of the gridpoints
                   (rows) in the table.  Spacing of grid values along an axis
                   may be uneven, but the full grid is required to a be a
                   regular, n-dimensional rectangular grid.  There is a column
                   of fluxes for each modelled emission line.  Model fluxes will
                   be normalised by NebulaBayes (see "norm_line" keyword below).
                   Any non-finite fluxes (e.g. nans) will be set to zero.
        grid_params: List of the unique names of the grid parameters as strings.
                     This list sets the order of the grid dimensions, i.e. the
                     order in which arrays in NebulaBayes will be indexed.  The
                     names must each match a column header in grid_file.
        lines_list: List of column names from grid_file corresponding to the
                    emission lines we'll be using in this NB_Model instance.

        Optional additional keyword arguments:
        interpd_grid_shape: A tuple of integers giving the size of each
                        dimension of the interpolated flux grids.  The order of
                        the integers corresponds to the order of parameters in
                        grid_params.  The default is 15 gridpoints along each
                        dimension.  These values have a major impact on the
                        speed of the grid interpolation.
        grid_error:     The systematic relative error on grid fluxes, as a
                        linear proportion.  Default is 0.35 (average of errors
                        of 0.15 dex above and below).
        """
        # Initialise and do some checks...
        print("Initialising NebulaBayes model...")

        # Determine if a custom interpolated grid shape was specified:
        interpd_grid_shape = kwargs.pop("interpd_grid_shape", [15]*len(grid_params))
        if len(interpd_grid_shape) != len(grid_params):
            raise ValueError("interpd_grid_shape should contain exactly "
                             "one integer for each parameter" )

        grid_rel_error = kwargs.pop("grid_error", 0.35)
        assert 0 < grid_rel_error < 1

        # Are there any remaining keyword arguments that weren't used?
        if len(kwargs) > 0:
            raise ValueError("Unknown keyword argument(s) " +
                                      ", ".join(str(k) for k in kwargs.keys()) )

        # Call grid initialisation:
        Raw_grids, Interpd_grids = NB1_Grid_working.initialise_grids(grid_file,
                                    grid_params, lines_list, interpd_grid_shape)
        Raw_grids.grid_rel_error = grid_rel_error
        Interpd_grids.grid_rel_error = grid_rel_error
        self.Raw_grids = Raw_grids
        self.Interpd_grids = Interpd_grids



    def __call__(self, obs_fluxes, obs_flux_errors, obs_emission_lines, **kwargs):
        """
        Run NebulaBayes Bayesian parameter estimation using the grids stored in
        this NB_Model object.
        Required positional arguments:
        obs_fluxes:         list or array of observed emission-line fluxes
        obs_flux_errors:    list or array of corresponding measurement errors
        obs_emission_lines: list of corresponding emission line names as strings
                            (must match names in header of input grid flux table)
        
        Optional keyword arguments which affect the parameter estimation:
        norm_line:     Both observed and grid fluxes will be normalised to this
                       line.  Because the likelihood calculation will use fluxes
                       that are actually ratios to this line, the choice may
                       affect parameter estimation.  Default: "Hbeta"
        deredden:      De-redden observed fluxes to match the Balmer decrement
                       at each interpolated grid point?  Default False.
        obs_wavelengths: If deredden=True, you must also supply a list of
                       wavelengths (Angstroems) associated with obs_fluxes.
                       Default: None
        prior:  The prior to use when calculating the posterior.  Either a
                user-defined function, the string "Uniform", or a list of length
                at least one. The entries in the list are tuples such as
                ("SII6716","SII6731") to indicate a line ratio to use as a prior.
                The listed line-ratio priors will all be multiplied together
                (weighted equally) and then normalised before being used in
                Bayes' Theorem.  See the code file "src/NB2_Prior.py" for the
                details of the prior calculations, including to see the required
                inputs and output for a user-defined prior function.
                Default: "Uniform"
        
        Optional additional keyword arguments regarding outputs (an output is
        only produced if a value is provided for the relevant keyword):
        param_display_names:  A dictionary of parameter display names for grid
                          parameters, for plotting purposes.  The dictionary
                          keys are parameter names from the grid file, and the
                          corresponding values are the "display" names.  The
                          display names can use markup (e.g. r"$\alpha$") in
                          order to include e.g. Greek letters.  Not all of the
                          grid parameters need to be in param_display_names;
                          raw parameter names will be used by default.
        posterior_plot:   A filename for saving out a results image of 2D and
                          1D marginalised posterior pdfs. The image file type is
                          specified by the file extension.
        prior_plot:       A filename; as for posterior_plot but for the prior
        likelihood_plot:  A filename; as for posterior_plot but for the likelihood
        estimate_table:   A filename for a csv file providing Bayesian parameter
                          estimates for the grid parameters
        best_model_table: A filename for a csv file which will compare observed
                          and model fluxes at the point defined by the Bayesian
                          parameter estimates.
        table_on_plots:   Include a "best model" flux comparison table on the
                          corner plots? Default: True
        line_plot_dir:    A directory; Plots showing the ND PDFs for each line
                          (the PDFs which contribute to the likelihood) are
                          saved here.  Saving these plots is quite quite slow.

        Returns a NB_Result object (defined in src/NB3_Bayes.py), which contains
        the data relevant to the Bayesian parameter estimation.
        """
        print("Running NebulaBayes...")

        Raw_grids = self.Raw_grids
        Interpd_grids = self.Interpd_grids

        norm_line = kwargs.pop("norm_line", "Hbeta") # Default "Hbeta"
        deredden = kwargs.pop("deredden", False) # Default False
        assert isinstance(deredden, bool)
        obs_wavelengths = kwargs.pop("obs_wavelengths", None) # Default None
        if deredden and (obs_wavelengths is None):
            raise ValueError("Must supply obs_wavelengths if deredden==True")
        if (obs_wavelengths is not None) and not deredden:
            pass # obs_wavelengths is unnecessary but will be checked anyway.
        # Process the input observed data; DF_obs is a pandas DataFrame table
        # where the emission line names index the rows:
        DF_obs = process_observed_data(obs_fluxes, obs_flux_errors,
                       obs_emission_lines, obs_wavelengths, norm_line=norm_line)

        input_prior = kwargs.pop("prior", "Uniform") # Default "Uniform"

        #----------------------------------------------------------------------
        # Handle options for NebulaBayes outputs:
        # Determine the list of parameter display names to use for plotting:
        param_display_names = list(Interpd_grids.param_names) # Default
        if "param_display_names" in kwargs:
            custom_display_names = kwargs.pop("param_display_names")
            for i, custom_name in enumerate(custom_display_names):
                param_display_names[i] = custom_name # Override default
        Interpd_grids.param_display_names = param_display_names
        # Include text "best model" table on posterior corner plots?
        table_on_plots = kwargs.pop("table_on_plots", True) # Default True
        # Filenames for output corner plot images?  Default None (no plotting)
        likelihood_plot = kwargs.pop("likelihood_plot", None)
        prior_plot      = kwargs.pop("prior_plot",      None)
        posterior_plot  = kwargs.pop("posterior_plot",  None)
        # Directory name for individual line PDF plots?
        line_plot_dir  = kwargs.pop("line_plot_dir", None)

        # Filenames for output csv tables?  Default None (don't write out table)
        estimate_table   = kwargs.pop("estimate_table",   None)
        best_model_table = kwargs.pop("best_model_table", None)

        # Are there any remaining keyword arguments that weren't used?
        if len(kwargs) > 0:
            raise ValueError("Unknown keyword argument(s): " +
                             ", ".join("'{0}'".format(k) for k in kwargs.keys()))

        #----------------------------------------------------------------------
        # Creat a ND_PDF_Plotter instance to plot corner plots
        ND_PDF_Plotter_1 = ND_PDF_Plotter(Raw_grids.paramName2paramValueArr)
        # Create a "NB_Result" object instance, which involves calculating
        # the prior, likelihood and posterior, along with parameter estimates:
        Result = NB3_Bayes.NB_Result(Interpd_grids, DF_obs, ND_PDF_Plotter_1,
                           norm_line=norm_line, deredden=deredden, 
                           input_prior=input_prior, line_plot_dir=line_plot_dir)

        #----------------------------------------------------------------------
        # Outputs
        if estimate_table is not None: # Save parameter estimates table?
            Result.Posterior.DF_estimates.to_csv(estimate_table, index=True,
                                                            float_format='%.5f')
        if best_model_table is not None: # Save flux comparison table?
            Result.Posterior.DF_best.to_csv(best_model_table, index=True,
                                                            float_format='%.5f')

        # Plot corner plots if requested:
        pdf_map = { "likelihood" : (likelihood_plot, Result.Likelihood),
                    "prior"      : (prior_plot,      Result.Prior     ),
                    "posterior"  : (posterior_plot,  Result.Posterior )  }
        for pdf_name, (out_image_name, NB_nd_pdf) in pdf_map.items():
            if out_image_name == None:
                continue # Only do plotting if an image name was specified:
            if table_on_plots is True: # Include a fixed-width text table on image
                pd.set_option("display.precision", 4)
                plot_anno = ("Observed fluxes vs. model fluxes at the gridpoint of"
                             "\nparameter best estimates in the "+pdf_name+"\n")
                plot_anno += str(NB_nd_pdf.DF_best) + "\n\n"
                plot_anno += r"$\chi^2_r$ = {0:.1f}".format(NB_nd_pdf.chi2)
            else:
                plot_anno = None
            NB_nd_pdf.Grid_spec.param_display_names = param_display_names
            print("Plotting corner plot for the", pdf_name, "...")
            Result.Plotter(NB_nd_pdf, out_image_name, plot_anno)

        
        print("NebulaBayes finished.")
        return Result




def process_observed_data(obs_fluxes, obs_flux_errors, obs_emission_lines,
                                                    obs_wavelengths, norm_line):
    """
    Error-check the input observed emission line data, form it into a pandas
    DataFrame table, and normalise by the specified line.
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
    if obs_wavelengths is not None: # Leave out of DataFrame if not provided
        obs_dict["Wavelength"] = obs_wavelengths
    obs_dict["Flux"] = obs_fluxes
    obs_dict["Flux_err"] = obs_flux_errors
    DF_obs = pd.DataFrame(obs_dict)
    DF_obs.set_index("Line", inplace=True) # Row index is the emission line name

    # Normalise the fluxes:
    if norm_line not in DF_obs.index.values:
        raise ValueError("'norm_line' {0} not found in input line names".format(
                                                                     norm_line))
    norm_flux = DF_obs.loc[norm_line, "Flux"] * 1.0
    DF_obs["Flux"] = DF_obs["Flux"].values / norm_flux
    DF_obs["Flux_err"] = DF_obs["Flux_err"].values / norm_flux
    assert np.isclose(DF_obs.loc[norm_line, "Flux"], 1.0) # Ensure normalised

    return DF_obs


