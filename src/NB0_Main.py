from __future__ import print_function, division
from collections import OrderedDict as OD
import numpy as np  # Core numerical library
import pandas as pd # For tables ("DataFrame"s)
from . import NB1_Process_grids
from . import NB3_Bayes
from .NB4_Plotting import Plot_Config, _make_plot_annotation, ND_PDF_Plotter
from .._version import __version__



"""
NebulaBayes
Adam D. Thomas
Research School of Astronomy and Astrophysics
Australian National University
2015 - 2017

The NB_Model class in this module is the entry point for performing Bayesian
parameter estimation.  The data are a set of emission line flux measurements
with associated errors.  The model is a photoionisation model, varied in a grid
over n=2 or more parameters, input as n-dimensional grids of fluxes for each
emission line.  The model is for an HII region or AGN Narrow Line Region, for
example.
The measured and modelled emission line fluxes are compared to calculate a
"likelihood" probability distribution, before Bayes' Theorem is applied to
produce an n-dimensional "posterior" probability distribution for the values of
the parameters.  The parameter values are estimated from 1D marginalised
posteriors.

NebulaBayes is heavily based on IZI (Blanc+2015).
"""



class NB_Model(object):
    """
    Primary class for working with NebulaBayes.  To use, initialise a class
    instance with a model grid and then call the instance one or more times to
    run Bayesian parameter estimation.
    """

    def __init__(self, grid_table, grid_params=None, lines_list=None, **kwargs):
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
            ["log U", "log P/k", "12 + log O/H", "E_peak"] for
            grid_table == "NLR".  Permute the list to change the order of the
            grid dimensions, i.e. the order of array indexing in NebulaBayes
            and the order of parameters in the output plots.
        lines_list : list of strings or None, optional
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
        grid_error : float between 0 and 1, optional
            The systematic relative error on grid fluxes, as a linear
            proportion.  Default is 0.35 (average of errors 0.15 dex above and
            0.15 dex below).

        Returns
        -------
        An NB_Model instance with attributes Interpd_grids, Raw_grids and
        Plotter. 
        """
        print("Initialising NebulaBayes (v{0}) model...".format(__version__))

        if grid_params is None:
            # Note that grid_table is validated when loading the table
            if isinstance(grid_table, str) and grid_table in ["HII", "NLR"]:
                if grid_table == "HII":
                    grid_params = ["log U", "log P/k", "12 + log O/H"]
                elif grid_table == "NLR":
                    grid_params = ["log U", "log P/k", "12 + log O/H", "E_peak"]
            else:
                raise ValueError("grid_params must be specified unless "
                                 " grid_table is 'HII' or 'NLR'")

        n_params = len(grid_params)
        if len(set(grid_params)) != n_params: # Parameter names non-unique?
            raise ValueError("grid_params are not all unique")

        # If lines_list isn't specified it'll be created when loading the grid
        if lines_list is not None:
            if len(set(lines_list)) != len(lines_list): # Lines non-unique?
                raise ValueError("Line names in lines_list are not all unique")
            if len(lines_list) < 2:
                raise ValueError("At least two modelled lines are required "
                                 "(one is for normalising)")

        # Interpolated grid shape
        default_shape = [int(6e4**(1./n_params))] * n_params  # 6e4 pts total
        interpd_grid_shape = kwargs.pop("interpd_grid_shape", default_shape)
        if len(interpd_grid_shape) != n_params:
            raise ValueError("Bad length for interpd_grid_shape: needs length" +
                             " {0} (the number of parameters)".format(n_params))

        grid_rel_error = kwargs.pop("grid_error", 0.35)  # Default: 0.35
        if not 0 < grid_rel_error < 1:
            raise ValueError("grid_error must be between 0 and 1")

        # Are there any remaining keyword arguments that weren't used?
        if len(kwargs) > 0:
            raise ValueError("Unknown keyword argument(s) " +
                                      ", ".join(str(k) for k in kwargs.keys()) )

        # Call grid initialisation:
        Raw_grids, Interpd_grids = NB1_Process_grids.initialise_grids(grid_table,
                                    grid_params, lines_list, interpd_grid_shape)
        Raw_grids.grid_rel_error = grid_rel_error
        Interpd_grids.grid_rel_error = grid_rel_error
        self.Raw_grids = Raw_grids
        self.Interpd_grids = Interpd_grids
        # Creat a ND_PDF_Plotter instance to plot corner plots
        self.Plotter = ND_PDF_Plotter(Raw_grids.paramName2paramValueArr)
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
            (-np.inf) to indicate that the measurement is an upper limit.
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
        deredden : bool
            De-redden observed fluxes to match the Balmer decrement at each
            interpolated grid point?  Only supported for norm_line == "Hbeta",
            when "Halpha" is also supplied, and if there are no upper bounds in
            the measurements.  Default: False
        obs_wavelengths : list of floats
            A list of wavelengths (Angstroems) corresponding to obs_line_names,
            which must be supplied if deredden == True.  Default: None
        prior : list of ("line1","line2") tuples, or "Uniform", or a callable.
            The prior to use when calculating the posterior.  Either a user-
            defined function, the string "Uniform", or a list of length at least
            one. Entries in the list are tuples such as ("SII6716","SII6731")
            to specify a line ratio to use as a prior.  The listed line-ratio
            priors will all be multiplied together (weighted equally) and then
            normalised before being used in Bayes' Theorem.  See the code file
            "src/NB2_Prior.py" for the details of the prior calculations,
            including to see the required inputs and output for a user-defined
            prior function.  Default: "Uniform"

        Optional parameters - outputs
        -----------------------------
        Provide a value for a keyword to produce the corresponding output.
        param_display_names : dict
            Display names for grid parameters, for plotting purposes.  The
            dictionary keys are parameter names from grid_params, and the
            corresponding values are the "display" names.  The display names can
            include markup (r"$\alpha$") to include e.g. Greek letters.  Raw
            text names will be used where a display name is not provided.
        posterior_plot : str
            A filename for a 'corner' plot of 1D and 2D marginalised posterior
            PDFs. The image file type is specified by the file extension.
        prior_plot : str
            As for posterior_plot but for the prior
        likelihood_plot : str
            As for posterior_plot but for the likelihood
        estimate_table : str
            A filename for a csv file containing Bayesian parameter estimates
            for the grid parameters
        best_model_table : str
            A filename for a csv file which will compare observed and model
            fluxes at the point defined by the Bayesian parameter estimates.
        line_plot_dir : str
            A directory where 'corner' plots showing the nD PDFs for each line
            will be saved.  These PDFs are the contribution to the likelihood
            from each line, and the plots show the constraints each line
            provides.  Saving these plots is slow; a ".pdf" file type is used.
        plot_configs : list of dicts
            A list of four dictionaries which update plotting options for the
            0) Prior, 1) Likelihood, 2) Posterior, and 3) Individual line plots.
            To use the same config dict object for all four plot types, write
            "plot_configs=[my_dict]*4".  Valid items in each dictionary are:
                table_on_plots : bool
                    Include a text "best model" flux comparison table on the
                    'corner' plots?  (Doesn't apply to individual line plots).
                    Default: True
                show_legend : bool
                    Show the legend? (Doesn't apply to 1D grids). Default: True
                cmap : str cmap name or matplotlib.colors.Colormap instance
                    The colormap for the images of the 2D marginalised PDFs.
                    Default: From black to white through green
                callback : callable
                    A function called instead of saving the plot, which may be
                    used to save a customised figure (e.g. with custom
                    annotations).

        Returns
        -------
        NB_Result : src.NB3_Bayes.NB_Result instance
            An object (class defined in src/NB3_Bayes.py) that contains the
            data relevant to the Bayesian parameter estimation.  Attributes:
            DF_obs : pandas DataFrame
                Contains the observed emission line fluxes, errors, and
                wavelengths (if provided), with a row for each line.
            Grid_spec : src.NB1_Process_grids.Grid_description instance
                Contains attributes describing the interpolated grids,
                including lists of parameter names and values, and mappings to
                and from indices along the dimensions.
            Likelihood : src.NB3_Bayes.NB_nd_pdf instance
                Object which holds important information...
            Plot_Config : src.NB4_Plotting.Plot_Config instance

            Plotter : src.NB4_Plotting.ND_PDF_Plotter instance

            Posterior : src.NB3_Bayes.NB_nd_pdf instance
                As for the Likelihood attribute, but for the posterior PDF
            Prior : src.NB3_Bayes.NB_nd_pdf instance
                As for the Likelihood attribute, but for the prior PDF
            deredden : bool
                Value of the "deredden" keyword.  Were observed fluxes
                dereddened all over the parameter space?
            obs_flux_arrs : list of numpy arrays

            obs_flux_err_arrs : list of numpy arrays

        """
        print("Running NebulaBayes...")

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
        deredden = kwargs.pop("deredden", False) # Default False
        assert isinstance(deredden, bool)
        if deredden and not all((l in obs_line_names) for l in ["Halpha","Hbeta"]):
            raise ValueError("'Halpha' and 'Hbeta' must be provided for deredden=True")
        obs_wavelengths = kwargs.pop("obs_wavelengths", None) # Default None
        if deredden and (obs_wavelengths is None):
            raise ValueError("Must supply obs_wavelengths for deredden=True")
        if deredden is False and (obs_wavelengths is not None):
            pass # obs_wavelengths is unnecessary but will be checked anyway.
        # Process the input observed data; DF_obs is a pandas DataFrame table
        # where the emission line names index the rows:
        DF_obs = _process_observed_data(obs_fluxes, obs_flux_errors,
                       obs_line_names, obs_wavelengths, norm_line=norm_line)
        for line in DF_obs.index:  # Check observed emission lines are in grid
            if line not in self.Interpd_grids.grids["No_norm"]:
                raise ValueError("The line {0}".format(line) + 
                                 " was not previously loaded from grid table")

        input_prior = kwargs.pop("prior", "Uniform") # Default "Uniform"

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

        # Any remaining keyword arguments that weren't used?
        if len(kwargs) > 0:
            raise ValueError("Unknown keyword argument(s): " +
                             ", ".join("'{0}'".format(k) for k in kwargs.keys()))

        #----------------------------------------------------------------------
        # Create a "NB_Result" object instance, which involves calculating
        # the prior, likelihood and posterior, along with parameter estimates:
        Result = NB3_Bayes.NB_Result(self.Interpd_grids, DF_obs, self.Plotter,
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
            print("Plotting corner plot for the", ndpdf_name.lower(), "...")
            self.Plotter(NB_nd_pdf, out_image_name, config=Plot_Config_1)

        self.Plot_Config = Plot_Config_1
        print("NebulaBayes finished.")
        return Result



def _process_observed_data(obs_fluxes, obs_flux_errors, obs_line_names,
                                                    obs_wavelengths, norm_line):
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
        if np.any(~np.isfinite(obs_wavelengths)): # Any non-finite?
            raise ValueError("An emission line wavelength isn't finite")
        if np.any(obs_wavelengths <= 0): # Any non-positive?
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

    # Form the data from the observations into a pandas DataFrame table.
    obs_dict = OD([("Line", obs_line_names)])
    if obs_wavelengths is not None: # Leave out of DataFrame if not provided
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


