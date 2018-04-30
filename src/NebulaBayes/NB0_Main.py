from __future__ import print_function, division
from collections import OrderedDict as OD
import logging

import numpy as np  # Core numerical library
import pandas as pd # For tables ("DataFrame"s)
from . import NB1_Process_grids
from . import NB3_Bayes
from .NB4_Plotting import Plot_Config, _make_plot_annotation, ND_PDF_Plotter
from ._version import __version__
from ._compat import _str_type  # Compatibility



class NB_Model(object):
    """
    Primary class for working with NebulaBayes.  To use, initialise a class
    instance with a model grid and then call the instance one or more times to
    run Bayesian parameter estimation.
    """

    def __init__(self, grid_table, grid_params=None, line_list=None, **kwargs):
        """
        Initialise an instance of the NB_Model class.  Load the model fluxes
        and interpolate them to a higher-density grid, ready to calculate
        likelihoods when this NB_Model instance is called with observed fluxes.

        Basic parameters
        ----------
        grid_table : "HII", "NLR", str filename, or a pandas DataFrame
            The table of photoionisation model grid fluxes.  Allowed inputs are
            the strings "HII" or "NLR" to use one of the NebulaBayes built-in
            grids, a filename for a csv, FITS (.fits) or compressed FITS
            (fits.gz) file, or a pre-loaded pandas DataFrame instance.
            Each gridpoint (point in parameter space) is a row in the table.
            There is a column for each parameter; the location of a gridpoint
            is defined by these parameter values.  There is a column of fluxes
            for each modelled emission line.
            No assumptions are made about the order of the gridpoints (rows) or
            the order of the columns in the table.  Unnecessary columns will be
            ignored, but the number of rows must be exact - the grid must be
            rectangular with exactly one gridpoint for every possible
            combination of included parameter values.  Note that the sampling
            of a parameter (spacing of gridpoints) may be uneven.
            Model fluxes will be normalised by NebulaBayes (see "norm_line"
            parameter to __call__ below).  Any non-finite fluxes (e.g. NaNs)
            will be set to zero.
        grid_params : list of strings or None, optional
            The names of the grid parameters, which must be specified if a grid
            other than the "HII" or "NLR" built-in grids is used.  Each name
            must match a column header in the grid_table.  Default lists are
            ["log U", "log P/k", "12 + log O/H"] for grid_table == "HII", and
            ["log U", "log P/k", "12 + log O/H", "log E_peak"] for
            grid_table == "NLR".  Permute the list to change the order of the
            grid dimensions, i.e. the order of array indexing in NebulaBayes
            and the order of parameters in the output plots.
        line_list : list of strings or None, optional
            The emission lines to use in this NB_Model instance.  Each name
            must match a column name in grid_table.  Exclude lines which won't
            be used in parameter estimation to save time and memory.  By
            default all non-parameter table columns are included.

        Optional parameters
        -------------------
        interpd_grid_shape : list of integers, optional
            The size of each dimension of the interpolated flux grids - larger
            numbers give a higher density uniform sampling of the fluxes.  The
            order of the integers corresponds to the order of parameters in
            grid_params.  The default is to have an equal number of points
            along each dimension such that there are ~60000 points in total in
            an interpolated grid.  The interpd_grid_shape has a major impact on
            the speed of the grid interpolation and on memory usage.
        interp_order : integer, optional
            The order of the polynomials used for interpolating model fluxes,
            either 1 (linear) or 3 (cubic; an experimental option).  Default: 1
        grid_error : float between 0 and 1, optional
            The systematic relative error on grid fluxes, as a linear
            proportion.  Default is 0.35 (average of errors 0.15 dex above and
            0.15 dex below).

        Returns
        -------
        An NB_Model instance with the following two public attributes:
        Interpd_grids : NB1_Process_grids.NB_Grid instance
            Object which holds the interpolated model line fluxes in the
            "grids" attribute.  A flux array can be accessed with e.g.
            Interpd_grids.grids["No_norm"]["OIII5007"] (for the un-normalised
            grids).  Other attributes include "param_names",
            "param_values_arrs" and "shape", which are in the order
            corresponding to the indexing of the flux arrays, "ndim" and
            "n_gridpoints".
        Raw_grids : NB1_Process_grids.NB_Grid instance
            As for Interpd_grids, but holds the flux arrays corresponding to
            the input grid table, before interpolation.  Arrays are
            accessed as e.g. Raw_grids.grids["OIII5007"]
        """
        NB_logger.info("Initialising NebulaBayes (v{0}) model...".format(
                                                                  __version__))

        if grid_params is None:
            # Note that grid_table is validated when loading the table
            if isinstance(grid_table,_str_type) and grid_table in ["HII","NLR"]:
                if grid_table == "HII":
                    grid_params = ["log U", "log P/k", "12 + log O/H"]
                elif grid_table == "NLR":
                    grid_params = ["log U", "log P/k", "12 + log O/H",
                                   "log E_peak"]
            else:
                raise ValueError("grid_params must be specified unless "
                                 "grid_table is 'HII' or 'NLR'")

        n_params = len(grid_params)
        if len(set(grid_params)) != n_params: # Parameter names non-unique?
            raise ValueError("grid_params are not all unique")

        # If line_list isn't specified it'll be created when loading the grid
        if line_list is not None:
            if len(set(line_list)) != len(line_list):  # Lines non-unique?
                raise ValueError("Line names in line_list are not all unique")
            if len(line_list) < 2:
                raise ValueError("At least two modelled lines are required "
                                 "(one is for normalising)")

        # Interpolated grid shape
        default_shape = [int(6e4**(1./n_params))] * n_params  # 6e4 pts total
        interpd_grid_shape = kwargs.pop("interpd_grid_shape", default_shape)
        if len(interpd_grid_shape) != n_params:
            raise ValueError("Bad length for interpd_grid_shape: needs length" +
                             " {0} (the number of parameters)".format(n_params))

        interp_order = kwargs.pop("interp_order", 1)  # Default: 1 (linear)
        if not interp_order in [1, 3]:
            raise ValueError("interp_order must be either 1 or 3")

        grid_rel_error = kwargs.pop("grid_error", 0.35)  # Default: 0.35
        if not 0 <= grid_rel_error < 1:
            raise ValueError("grid_error must be between 0 and 1")

        # Are there any remaining keyword arguments that weren't used?
        if len(kwargs) > 0:
            raise ValueError("Unknown keyword argument(s) " +
                                      ", ".join(str(k) for k in kwargs.keys()) )

        # Call grid initialisation:
        Raw_grids, Interpd_grids = NB1_Process_grids.initialise_grids(grid_table,
                                    grid_params, line_list, interpd_grid_shape,
                                    interp_order=interp_order)
        Raw_grids.grid_rel_error = grid_rel_error
        Interpd_grids.grid_rel_error = grid_rel_error
        self.Raw_grids = Raw_grids
        self.Interpd_grids = Interpd_grids
        # Creat an ND_PDF_Plotter instance to plot corner plots
        self._Plotter = ND_PDF_Plotter(Raw_grids.paramName2paramValueArr)
        # The ND_PDF_Plotter instance will be an attribute on both this
        # NB_Model instance and on all "NB_Result" instances created later.



    def __call__(self, obs_fluxes, obs_flux_errors, obs_line_names, **kwargs):
        """
        Run NebulaBayes Bayesian parameter estimation using the interpolated
        grids stored in this NB_Model object.
        
        Parameters
        ----------
        obs_fluxes : list of floats
            The observed emission-line fluxes.  Use a flux of minus infinity
            (-np.inf) for a measurement that is an upper limit.  Fluxes and
            errors must already be dereddened, unless "deredden" below is True.
        obs_flux_errors : list of floats
            The corresponding measurement errors
        obs_line_names : list of str
            The corresponding emission line names, matching names in the header
            of the input grid flux table
        
        Optional parameters - for parameter estimation
        ----------------------------------------------
        norm_line : str
            Observed and grid fluxes will be normalised to this emission line.
            Because the likelihood calculation will use fluxes that are
            actually ratios to this line, the choice may affect parameter
            estimation.  Where the interpolated model grid for norm_line has
            value zero, the normalised grids are set to zero.  Default: "Hbeta"
        likelihood_lines : list of str
            A subset of obs_line_names that specifies the lines to include in
            the likelihood calculation.  By default, all the obs_line_names are
            used.  Lines that appear in obs_line_names but aren't used in the
            likelihood are included in the best_model_table (see below), and
            may be used in line ratio priors (see below).
        deredden : bool
            De-redden observed fluxes to match the Balmer decrement at each
            interpolated grid point?  Only supported if norm_line == "Hbeta",
            "Halpha" is also supplied, and there are no upper bounds in the
            measurements.  Default: False
        obs_wavelengths : list of floats
            A list of wavelengths (Angstroems) corresponding to obs_line_names,
            which must be supplied if deredden == True.  Default: None
        prior : list of ("line1","line2") tuples, or "Uniform", or a callable,
            or a numpy array.
            The prior to use in Bayes' Theorem when calculating the posterior.
            Either the string "Uniform", a user-defined callback function, a
            numpy array over the whole parameter space (with shape matching
            interpd_grid_shape), or a list of length at least one.  Entries in
            the list are tuples such as ("SII6716","SII6731") to specify a line
            ratio to use as a prior.  The listed line-ratio priors will all be
            multiplied together (weighted equally) and then normalised before
            being used in Bayes' Theorem.  See the "docs/Example-advanced.py"
            file to see an example user-defined callback function, showing the
            arguments it must accept.
            Default: "Uniform"

        Optional parameters - outputs
        -----------------------------
        Provide a value for a keyword to produce the corresponding output.
        estimate_table : str
            A filename for a csv file containing Bayesian parameter estimates
            for the grid parameters
        posterior_plot : str
            A filename for a 'corner' plot of 1D and 2D marginalised posterior
            PDFs. The image file type is specified by the file extension.
        prior_plot : str
            As for posterior_plot but for the prior
        likelihood_plot : str
            As for posterior_plot but for the likelihood
        best_model_table : str
            A filename for a csv file which will compare observed and model
            fluxes at the point defined by the Bayesian parameter estimates.
        line_plot_dir : str
            A directory where 'corner' plots showing the nD PDFs for each line
            will be saved.  These PDFs are the contribution to the likelihood
            from each line, and the plots show the constraints each line
            provides.  Saving these plots is slow; a ".pdf" file type is used.
        param_display_names : dict
            Display names for grid parameters, for plotting purposes.  The
            dictionary keys are parameter names from grid_params, and the
            corresponding values are the "display" names.  The display names
            can include markup (r"$\\alpha$") to include e.g. Greek letters.
            Raw text names will be used where a display name is not provided.
        plot_configs : list of dicts
            A list of four dictionaries which update plotting options for the
            0) Prior, 1) Likelihood, 2) Posterior, and 3) Individual line plots.
            To use the same config dict object for all four plot types, write
            "plot_configs=[my_dict]*4".  Valid items in each dictionary are:
                table_on_plot : bool
                    Include a text "best model" flux comparison table on the
                    'corner' plot?  (Doesn't apply to individual line plots).
                    Default: True
                show_legend : bool
                    Show the legend? (Doesn't apply to 1D grids). Default: True
                legend_fontsize : float
                    Fontsize for label text in legend.  Default: 4.5 pts
                cmap : str cmap name or matplotlib.colors.Colormap instance
                    The colormap for the images of the 2D marginalised PDFs.
                    Default: From black to white through green
                callback : callable
                    A user-defined function called instead of saving the plot,
                    which may be used to save a customised figure (e.g. with
                    annotations).  See the "docs/Example-advanced.py" file for
                    an example function, showing the arguments it must accept.
        verbosity : str
            Determine how much information is printed to the terminal by
            setting the level of the NebulaBayes logger.  Allowed levels (in
            order of more to less output) are "DEBUG", "INFO" and "WARNING".
            The logger object may be accessed as NebulaBayes.NB_logger.
            Default: "DEBUG"


        Returns
        -------
        NB_Result : NB3_Bayes.NB_Result instance
        This object holds the working and results of the Bayesian parameter
        estimation.  It has the following attributes, listed here roughly in
        order of decreasing importance:
            Posterior : NB3_Bayes.NB_nd_pdf instance
                Holds data for the posterior PDF, and has these attributes:
                DF_estimates : pandas DataFrame table holding the parameter
                    estimates, limits of the 68\% and 95\% credible intervals
                    (CIs), and results of checks for the PDF being up against
                    the edge of the parameter space or having a pathological
                    shape.  The CIs are calculated using the 1D CDF
                    independently of the parameter estimate.  The "low" and
                    "high" CI limits may be -inf or +inf respectively
                    if there is significant probability density near the edges
                    of the parameter space.  Because the parameter estimate is
                    at the peak of the PDF, it may not lie inside the CIs, and
                    "Est_in_CI68?" and "Est_in_CI95?" are checks for this.
                    There are also separate checks for whether the parameter
                    esimate is at the "lower" or "upper" bounds of the
                    parameter space, and for the number of local maxima in the
                    1D PDF.  This table may be accessed as
                    'NB_Result.Posterior.DF_estimates'.
                best_model : dict describing the model at the point in the PDF
                    defined by the parameter estimates, with the following keys
                    and values:
                    "table" : pandas DataFrame table comparing model and
                              observed fluxes for the "best model".  There is a
                              row per emission line.  Accessed as
                              'NB_Result.Posterior.best_model["table"]'
                    "chi2" :  The reduced chi^2 of the fit
                    "extinction_Av_mag" : The visual extinction in magnitudes
                              (if "deredden" was True)
                    "grid_location" : Indices for the interpolated grid that
                              define the point of the parameter estimates
                nd_pdf : The numpy ndarray which samples the posterior PDF over
                    the full interpolated parameter space
                Grid_spec : NB1_Process_grids.Grid_description instance which
                    has attributes "param_names", "param_values_arrs", "shape",
                    etc., describing the geometry of the interpolated grid.
                marginalised_1D : dict that maps parameter names to numpy
                                  arrays that sample the 1D marginalised PDFs
                marginalised_2D : dict mapping tuples of two parameter names to
                                  numpy arrays that sample 2D marginalised PDFs
                name : str giving the PDF name - "Posterior"
            Likelihood : NB3_Bayes.NB_nd_pdf instance
                As for the Posterior attribute, but for the likelihood PDF.
                The parameter estimates and best model are therefore for the
                "maximum likelihood estimates", rather than the full Bayesian
                parameter estimates.
            Prior : NB3_Bayes.NB_nd_pdf instance
                As for the Posterior and Likelihood attributes, but for the
                prior PDF.  The parameter estimates are for the "best" model
                according to the prior information, without updating for the
                observational data.
            deredden : bool
                Value of the "deredden" keyword.  Were observed fluxes
                dereddened all over the parameter space?
            DF_obs : pandas DataFrame
                Table of the observed emission line fluxes, errors, and
                wavelengths (if provided), with a row for each line.  Fluxes
                and errors are normalised to the norm_line flux.
            obs_flux_arrs : list of numpy arrays
                A dictionary mapping each line name to an array of the observed
                fluxes used at each point in the parameter space.  The fluxes
                will be the same everywhere unless deredden=True, in which case
                the fluxes were dereddened to match the predicted Balmer
                decrement at each point in the parameter space.
            obs_flux_err_arrs : list of numpy arrays
                Same as obs_flux_arrs, but for the observed flux errors.
            Grid_spec : NB1_Process_grids.Grid_description instance
                As for NB_Result.Posterior.Grid_spec
            Plotter : NB4_Plotting.ND_PDF_Plotter instance
                Object which is called with an NB3_Bayes.NB_nd_pdf object and
                an output filename to plot an nD PDF, e.g. the posterior.
            Plot_Config : NB4_Plotting.Plot_Config instance
                Object with a "configs" attribute, which is a dict storing
                options for each of the types of plots

        """
        if len(set(obs_line_names)) != len(obs_line_names):
            raise ValueError("obs_line_names are not all unique")
        line_names_upper = [l.upper() for l in obs_line_names]
        if "norm_line" not in kwargs and "HBETA" not in line_names_upper:
            raise ValueError("Can't normalise by default line 'Hbeta': not"
                             " found in obs_line_names.  Consider setting"
                             " keyword 'norm_line' to another line.")
        norm_line = kwargs.pop("norm_line", "Hbeta")  # Default "Hbeta"
        if norm_line not in obs_line_names:
            # We want norm_line to be case-insensitive.  If there is exactly
            # 1 case-insensitive match in obs_line_names, use the matched name.
            if line_names_upper.count(norm_line.upper()) == 1:
                index_nl = line_names_upper.index(norm_line.upper())
                norm_line = obs_line_names[index_nl]
            else:
                raise ValueError("norm_line '{0}'".format(norm_line) +
                                 " not found in obs_line_names")
        likelihood_lines = kwargs.pop("likelihood_lines", None)  # Default None
        deredden = kwargs.pop("deredden", False)  # Default False
        assert isinstance(deredden, bool)
        if deredden and not all((l in obs_line_names) for l in ["Halpha","Hbeta"]):
            raise ValueError("'Halpha' and 'Hbeta' must be provided for deredden==True")
        obs_wavelengths = kwargs.pop("obs_wavelengths", None)  # Default None
        if deredden and (obs_wavelengths is None):
            raise ValueError("Must supply obs_wavelengths for deredden==True")
        if deredden is False and (obs_wavelengths is not None):
            pass # obs_wavelengths is unnecessary but will be checked anyway.
        # Process the input observed data; DF_obs is a pandas DataFrame table
        # where the emission line names index the rows:
        DF_obs = _process_observed_data(obs_fluxes, obs_flux_errors,
                        obs_line_names, obs_wavelengths=obs_wavelengths,
                        norm_line=norm_line, likelihood_lines=likelihood_lines)
        for line in DF_obs.index:  # Check observed emission lines are in grid
            if line not in self.Interpd_grids.grids["No_norm"]:
                raise ValueError("The line {0}".format(line) + 
                                 " was not previously loaded from grid table")

        input_prior = kwargs.pop("prior", "Uniform")  # Default "Uniform"

        #----------------------------------------------------------------------
        # Handle options for NebulaBayes outputs:
        # Determine the parameter display names to use for plotting:
        param_list = list(self.Interpd_grids.param_names)
        param_display_names = OD(zip(param_list, param_list)) # Default; ordered
        if "param_display_names" in kwargs:
            custom_display_names = kwargs.pop("param_display_names")
            if not isinstance(custom_display_names, dict):
                raise TypeError("param_display_names must be a dict")
            for p, custom_name in custom_display_names.items():
                if p not in param_list:
                    raise ValueError("Unknown parameter in param_display_names")
                param_display_names[p] = custom_name  # Override default
        self.Interpd_grids.param_display_names = list(param_display_names.values())
        # Configuration for plots
        input_plot_configs = kwargs.pop("plot_configs", [{}]*4)
        Plot_Config_1 = Plot_Config(input_plot_configs)
        # Handle directory and file names for the outputs:
        output_locations = {}
        for key in ["likelihood_plot", "prior_plot", "posterior_plot",
                    "line_plot_dir", "estimate_table", "best_model_table"]:
            output_locations[key] = kwargs.pop(key, None)
            # Default None means "Don't produce the relevant output"

        default_verbosity = "DEBUG"
        verbosity = kwargs.pop("verbosity", default_verbosity)
        if verbosity not in ["DEBUG", "INFO", "WARNING"]:
            raise ValueError("verbosity must be 'DEBUG', 'INFO', or 'WARNING'")
        NB_logger.setLevel(verbosity)

        # Any remaining keyword arguments that weren't used?
        if len(kwargs) > 0:
            raise ValueError("Unknown keyword argument(s): " +
                             ", ".join("'{0}'".format(k) for k in kwargs.keys()))

        #----------------------------------------------------------------------
        NB_logger.info("Running NebulaBayes parameter estimation...")
        # Create a "NB_Result" object instance, which involves calculating
        # the prior, likelihood and posterior, along with parameter estimates:
        Result = NB3_Bayes.NB_Result(self.Interpd_grids, DF_obs, self._Plotter,
                                     Plot_Config=Plot_Config_1,
                                     input_prior=input_prior, deredden=deredden,
                                line_plot_dir=output_locations["line_plot_dir"])

        #----------------------------------------------------------------------
        # Write out the results
        table_map = {"estimate_table"  : Result.Posterior.DF_estimates,
                     "best_model_table": Result.Posterior.best_model["table"]}
        for table_name, DF in table_map.items():
            out_table_name = output_locations[table_name]
            if out_table_name is not None:
                DF.to_csv(out_table_name, index=True, float_format="%.5f")

        # Plot corner plots if requested:
        for NB_nd_pdf in [Result.Prior, Result.Likelihood, Result.Posterior]:
            ndpdf_name = NB_nd_pdf.name  # A "plot_type" from NB4_Plotting.py
            out_image_name = output_locations[ndpdf_name.lower() + "_plot"]
            if out_image_name is None:
                continue  # Only do plotting if an image name was specified
            # Add plot annotation to Plot_Config_1 ("table_for_plot" attribute)
            _make_plot_annotation(Plot_Config_1, NB_nd_pdf)
            NB_nd_pdf.Grid_spec.param_display_names = list(
                                                  param_display_names.values())
            NB_logger.info("Plotting corner plot for the {0}...".format(
                                                           ndpdf_name.lower()))
            self._Plotter(NB_nd_pdf, out_image_name, config=Plot_Config_1)

        NB_logger.info("NebulaBayes parameter estimation finished.")
        NB_logger.setLevel(default_verbosity)  # Reset
        return Result



def _process_observed_data(obs_fluxes, obs_flux_errors, obs_line_names,
                                obs_wavelengths, norm_line, likelihood_lines):
    """
    Error-check the input observed emission line data, form it into a pandas
    DataFrame table, and normalise by the specified line.

    Returns
    -------
    DF_obs : DataFrame
        Table of observed emission line data with a row for each emission line.
    """
    obs_fluxes = np.asarray(obs_fluxes, dtype=float)  # Ensure numpy array
    obs_flux_errors = np.asarray(obs_flux_errors, dtype=float)
    # Check measured data inputs:
    n_measured = len(obs_line_names)
    if (obs_fluxes.size != n_measured) or (obs_flux_errors.size != n_measured):    
        raise ValueError("Inputs obs_fluxes, obs_flux_errors and " 
                         "obs_line_names don't all have the same length.")
    if n_measured < 2:
        raise ValueError("At least two observed lines are required (one is for "
                                                                 "normalising)")
    if obs_wavelengths is not None:
        obs_wavelengths = np.asarray(obs_wavelengths, dtype=float)
        if obs_wavelengths.size != n_measured:
            raise ValueError("obs_wavelengths must have same length as obs_fluxes")
        # Check input wavelengths:
        if np.any(~np.isfinite(obs_wavelengths)):  # Any non-finite?
            raise ValueError("An emission line wavelength isn't finite")
        if np.any(obs_wavelengths <= 0):  # Any non-positive?
            raise ValueError("An emission line wavelength isn't positive")

    # Check input measured fluxes:
    if np.any(np.isnan(obs_fluxes)) or np.any(obs_fluxes == np.inf):
        # Any NaNs or +infs? (-inf is allowed, and means "upper bound")
        raise ValueError("A measured emission line flux is NaN or +inf")
    if np.any((obs_fluxes <= 0) & (obs_fluxes != -np.inf)):
        raise ValueError("A measured emission line flux isn't positive")
    
    # Check input measured flux errors:
    if np.any(~np.isfinite(obs_flux_errors)): # Any non-finite?
        raise ValueError("The flux error for an emission line isn't finite")
    if np.any(obs_flux_errors <= 0): # All positive?
        raise ValueError("The flux error for an emission line isn't positive")

    # Check likelihood_lines list:
    if likelihood_lines is None:
        likelihood_lines = obs_line_names
    if not all(isinstance(s, _str_type) for s in likelihood_lines):
        raise TypeError("All items in likelihood_lines must be strings")
    if len(likelihood_lines) < 2:
        raise ValueError("likelihood_lines list must have length at least 2")
    lines_diff = set(likelihood_lines) - set(obs_line_names)
    if len(lines_diff) > 0:
        raise ValueError("Lines in likelihood_lines not found in "
                         "obs_line_names: " + ", ".join(lines_diff))

    # Form the data from the observations into a pandas DataFrame table.
    obs_dict = OD([("Line", obs_line_names)])
    map_TF = {True: "Y", False: "N"}
    obs_dict["In_lhood?"] = [map_TF[l in likelihood_lines] for l in obs_line_names]
    if obs_wavelengths is not None:  # Leave out of DataFrame if not provided
        obs_dict["Wavelength"] = obs_wavelengths
    obs_dict["Flux"] = obs_fluxes
    obs_dict["Flux_err"] = obs_flux_errors
    DF_obs = pd.DataFrame(obs_dict)
    DF_obs.set_index("Line", inplace=True) # Row index is emission line name

    # Normalise the fluxes:
    norm_flux = DF_obs.loc[norm_line, "Flux"] * 1.0
    if norm_flux == 0:
        raise ValueError("The obs flux for norm_line ({0}) is 0".format(norm_line))
    DF_obs["Flux"] = DF_obs["Flux"].values / norm_flux
    DF_obs["Flux_err"] = DF_obs["Flux_err"].values / norm_flux
    assert np.isclose(DF_obs.loc[norm_line, "Flux"], 1.0) # Ensure normalised
    DF_obs.norm_line = norm_line  # Store as attribute on DataFrame
    # Note that storing metadata on DataFrames isn't trivial - we may lose the
    # "norm_line" attribute if we do some common operations on DF_obs.

    return DF_obs



def _configure_logger():
    """
    Create a logger for NebulaBayes so the user can easily control verbosity
    of the output.  We set the level of the "root" logger to debug.
    """
    NB_logger = logging.getLogger("NebulaBayes")
    Handler_1 = logging.StreamHandler()
    # Set the NB logger and the root logger to show all messages
    Handler_1.setLevel(logging.DEBUG)
    logging.getLogger("").setLevel(logging.DEBUG)  # For root logger
    # Set a format which works well for console output:
    Formatter_1 = logging.Formatter("%(message)s")
    Handler_1.setFormatter(Formatter_1)
    NB_logger.addHandler(Handler_1)
    return NB_logger

NB_logger = _configure_logger()


