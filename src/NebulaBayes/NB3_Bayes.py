from __future__ import print_function, division
from collections import OrderedDict as OD
import itertools    # For combinatorial combinations
import logging
import os.path

import numpy as np  # Core numerical library
import pandas as pd # For tables ("DataFrame"s)
from scipy.integrate import cumtrapz, simps
from scipy.signal import argrelextrema
from scipy.special import erf  # Error function
from .dereddening import deredden as do_dereddening
from .dereddening import Av_from_BD
from .NB1_Process_grids import Grid_description
from .NB2_Prior import calculate_prior


"""
Code to calculate the likelihood and posterior over an N-D grid, marginalise
pdfs to 1D and 2D marginalised pdfs, and generally do Bayesian parameter
estimation.
This module defines two custom NebulaBayes classes: NB_nd_pdf and NB_Result.

Adam D. Thomas 2015 - 2018
"""



NB_logger = logging.getLogger("NebulaBayes")



class CachedIntegrator(object):
    """
    Class to perform trapezoidal integration in arbitrary dimensions, one
    dimension at a time.  A cache is used to store previous results, which
    significantly speeds up finding all the 1D and 2D marginalised PDFs for
    grids with more than 3 dimensions.
    A custom class is used because functools.lru_cache is not available in
    the standard library in python 2 (and an extra dependency is avoided).
    """
    def __init__(self, full_nd_pdf, spacing):
        """
        Initialise CachedIntegrator instance with some data required for
        performing the integration.
        """
        self.spacing = spacing  # List holding the spacing (dx) for each axis.
        # The cache maps a tuple of the indices of the axes that have been
        # integrated out to the corresponding marginalised PDF array
        self.cache = {(): full_nd_pdf.copy()}


    def __call__(self, inds_already_integrated, ind_to_integrate):
        """
        Integrate out a single dimension from the PDF array, which may have
        already been marginalised over one or more dimensions.  Trapezoidal
        integration is used.  Previous integration results are cached, so they
        don't need to be recalculated.
        """
        assert ind_to_integrate not in inds_already_integrated
        assert isinstance(ind_to_integrate, int)
        assert isinstance(inds_already_integrated, list)  # List of ints
        cache_key = tuple(sorted(inds_already_integrated + [ind_to_integrate]))
        if cache_key in self.cache:
            return self.cache[cache_key]  # Return results of previous call

        start_arr = self.cache[tuple(sorted(inds_already_integrated))]
        # Integrate over the desired dimension/parameter/axis, using the
        # trapezoidal rule
        marginalized_arr = np.trapz(start_arr, axis=ind_to_integrate,
                                    dx=self.spacing[ind_to_integrate])
        self.cache[cache_key] = marginalized_arr  # Cache the results
        return marginalized_arr



class NB_nd_pdf(object):
    """
    Class to hold an N-dimensional PDF and its 1D and 2D marginalised forms.
    Methods on this class marginalise the PDF and perform parameter estimation.
    An instance of this class is used for each of the likelihood, prior and
    posterior.
    The actual nd_pdf array (an attribute of instances of this class) represents
    a probability distribution, i.e. dP = PDF(x) * dx, where the nd_pdf is an
    array of samples of the PDF.
    """
    def __init__(self, nd_pdf, NB_Result, Interpd_grids, name, DF_obs=None):
        """
        Initialise an instance of the NB_nd_pdf class.
        nd_pdf: A numpy ndarray holding the (linear) pdf.
        NB_Result: A NB_Result object (defined in this module)
        Interpd_grids: A NB_Grid object (defined in NB1_Process_grids.py)
                       holding the interpolated model grid fluxes and the
                       description of the model grid.
        DF_obs: A pandas DataFrame table holding the observed emission line
                fluxes and errors.  The "best model table" is only generated if
                this is supplied.
        The nd_pdf will be normalised.
        """
        if np.any(~np.isfinite(nd_pdf)):
            raise ValueError("The {0} nd_pdf is not entirely finite".format(name))
        if nd_pdf.min() < 0:
            raise ValueError("The {0} nd_pdf contains a negative value".format(name))
        self.nd_pdf = nd_pdf
        self.Grid_spec = NB_Result.Grid_spec
        self.name = name
        # The "name" must be one of the "plot_types" listed in NB4_Plotting.py
        # Add self.marginalised_2d and self.marginalised_1d attributes and
        # normalise the self.nd_pdf attribute:
        self._marginalise_pdf()
        # Make a parameter estimate table based on this nd_pdf
        self._make_parameter_estimate_table() # add self.DF_estimates attribute
        if DF_obs is not None:
            self.best_model = {}
            # For the "best" model, we calculate the following 3 items:
            # 1.) Make a table comparing the model and observed fluxes
            self._make_best_model_table(DF_obs, Interpd_grids, NB_Result)
                  # We added "table" to "best_model" dict
            # 2.) Calculate chi2 of the fit (add "chi2" to "best_model" dict):
            self._calculate_chi2(NB_Result.deredden, DF_obs)
            # 3.) Calculate implied extinction ("extinction_Av_mag" in dict):
            self._calculate_Av(NB_Result.deredden, Interpd_grids, DF_obs)



    def _marginalise_pdf(self):
        """
        Calculate normalised 1D and 2D marginalised pdfs for all possible
        combinations of parameters.  Store them as dictionaries in the
        attributes self.marginalised_2D and self.marginalised_1D.
        """
        # The interpolated grids have uniform spacing:
        spacing = [(v[1] - v[0]) for v in self.Grid_spec.param_values_arrs]
        n = self.Grid_spec.ndim
        #----------------------------------------------------------------------
        # Calculate the 2D marginalised pdf for every possible combination
        # of 2 parameters
        # List of all possible pairs of two parameter names:
        param_names = self.Grid_spec.param_names
        p2ind = self.Grid_spec.paramName2ind
        double_names = list(itertools.combinations(param_names, 2))
        self.Grid_spec.double_names = double_names
        # Looks something like: [('BBBP','Gamma'), ('BBBP','NT_frac'),
        #                        ('BBBP','UH_at_r_inner'), ('Gamma','NT_frac'),
        #                        ('Gamma','UH_at_r_inner'), ('NT_frac','UH_at_r_inner')]
        # Corresponding list of possible combinations of two parameter indices:
        double_indices = [(p2ind[p1], p2ind[p2]) for p1,p2 in double_names]
        self.Grid_spec.double_indices = double_indices
        # Looks something like: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        # Note that the order of indices in each tuple is from smaller to larger
        
        # Initialise dictionary of all possible 2D marginalised pdf arrays:
        marginalised_2D = {}  # The dict keys are tuples of two parameter names
        # Iterate over all possible pairs of parameters:
        Integrator = CachedIntegrator(self.nd_pdf, spacing)
        for double_name, param_inds_double in zip(double_names, double_indices):
            if self.nd_pdf.ndim == 2:
                marginalised_2D[double_name] = self.nd_pdf
                break  # We're already done!

            # Generate list of indices/dimensions/parameters to integrate over:
            inds_for_integration = np.arange(n).tolist()  # Initialise
            inds_for_integration.remove(param_inds_double[0])
            inds_for_integration.remove(param_inds_double[1])
            inds_for_integration.reverse() # Ensure we integrate over higher
            # dimensions first, so dimension index numbers are still valid
            # after each integration.

            # Integrate out the dimensions of the full array one dimension at a
            # time.  Keep integrating until the result only has two dimensions:
            integrated_inds = []
            for p_index in inds_for_integration:
                marginalised_array = Integrator(integrated_inds, p_index)
                integrated_inds.append(p_index)
            marginalised_2D[double_name] = marginalised_array  # Store result

        #----------------------------------------------------------------------
        # Calculate the 1D marginalised pdf for each individual parameter
        # Initialise dictionary of all 1D marginalised pdf arrays:
        marginalised_1D = {}

        # For the first parameter in param_names, integrate the first 2D
        # marginalised pdf over the other dimension (parameter), using
        # Simpson's rule:
        if len(marginalised_2D) > 0:  # If more than 1 dimension in grid
            integral_i = simps(marginalised_2D[double_names[0]], axis=1,
                                                                 dx=spacing[1])
            marginalised_1D[param_names[0]] = integral_i
            # np.trapz(marginalised_2D[double_names[0]], axis=1, dx=spacing[1])
        else:  # Must be a 1D grid
            marginalised_1D[param_names[0]] = self.nd_pdf

        # For all parameters after the first in param_names:
        for double_name, param_inds_double in zip(double_names[:n-1],
                                                  double_indices[:n-1]):
            # For each pair of parameters we take the second parameter, and 
            # integrate over the first parameter of the pair (which by
            # construction is always the first parameter in param_names).
            assert param_inds_double[0] == 0 
            param = param_names[param_inds_double[1]]
            # Integrate over first dimension (parameter) using Simpson's rule:
            marginalised_1D[param] = simps(marginalised_2D[double_name],
                                                         axis=0, dx=spacing[0])

        #----------------------------------------------------------------------
        # Calculate the 0D marginalised pdf (by which I mean find the
        # normalisation constant - the 0D marginalised pdf should be 1!), then
        # normalise all the PDFs.

        # Find the integral over all n dimensions.  We integrate over all the
        # 1D PDFs, and take the median of these values.
        integrals = [simps(pdf_1D, dx=spacing[0]) for
                                            pdf_1D in marginalised_1D.values()]
        integral = np.median(integrals)
        # integral = np.trapz(marginalised_1D[param_names[0]], dx=spacing[0])
        # Now normalise each 2D and 1D marginalised pdf and the full nD pdf.
        # In the case that all models in the grid are a poor fit to the data,
        # the likelihood may be zero everywhere.  In this case the integral
        # will be zero, and we don't normalise.
        if integral > 0:
            for double_name in double_names:
                # Divide arrays in-place in memory:
                marginalised_2D[double_name] /= integral
            for param in param_names:
                marginalised_1D[param] /= integral
            # Normalise the full pdf.  For the likelihood and prior, the
            # normalisation is important; for the posterior it's probably not.
            self.nd_pdf /= integral

        self.marginalised_2D = marginalised_2D
        self.marginalised_1D = marginalised_1D



    def _make_parameter_estimate_table(self):
        """
        Do parameter estimation for all parameters in the grid.  Where the
        nd_pdf is the posterior, this is the proper Bayesian parameter
        estimation.
        Creates a pandas DataFrame with a row for each parameter, stored as the
        attribute self.DF_estimates
        """
        # DataFrame columns and types, in order:
        columns = [("Parameter", np.str),  ("Estimate", np.float),
                   ("CI68_low", np.float), ("CI68_high", np.float),
                   ("CI95_low", np.float), ("CI95_high", np.float),
                   ("Est_in_CI68?", np.str),  ("Est_in_CI95?", np.str),
                   ("Est_at_lower?", np.str), ("Est_at_upper?", np.str),
                   ("P(lower)", np.float),    ("P(upper)", np.float),
                   ("P(lower)>50%?", np.str), ("P(upper)>50%?", np.str),
                   ("n_local_maxima", np.int)]
        n = self.Grid_spec.ndim
        OD1 = OD([(c, np.zeros(n, dtype=t)) for c,t in columns])
        DF_estimates = pd.DataFrame(OD1) # Initialise DataFrame
        DF_estimates.loc[:,"Parameter"] = self.Grid_spec.param_names
        DF_estimates.sort_values(by="Parameter", inplace=True)
        # Sort DF, so the DF is deterministic (note Upper and lower case param
        # names are sorted into separate sections)
        DF_estimates.set_index("Parameter", inplace=True)
        for col in [col for col,t in columns if t == np.float]:
            DF_estimates[col] = np.nan
        DF_estimates.loc[:,"n_local_maxima"] = -1
        
        # Fill in DF_estimates: 
        for p, pdf_1D in self.marginalised_1D.items():
            param_val_arr = self.Grid_spec.paramName2paramValueArr[p]
            p_dict = make_single_parameter_estimate(p, param_val_arr, pdf_1D)
            for field, value in p_dict.items():
                if field != "Parameter":
                    DF_estimates.set_value(p, field, value)

        self.DF_estimates = DF_estimates



    def _make_best_model_table(self, DF_obs, Interpd_grids, NB_Result):
        """
        Make a pandas dataframe comparing observed emission line fluxes with
        model fluxes for the model corresponding to the parameter estimates
        (the 'best' model).  The parameter estimates are derived from the 1D
        mariginalised PDFs, so note that this 'best' point in the parameter
        space does not necessarily correspond to the peak of any 2D
        marginalised pdf nor to any projection of the peak of the ND pdf to a
        lower parameter space.
        """
        DF_best = DF_obs.copy() # Index: "Line"; columns: "Flux", "Flux_err"
        # DF_obs may also possibly have a "Wavelength" column
        DF_best.rename(columns={"Flux":"Obs"}, inplace=True)

        inds_max = np.unravel_index(self.nd_pdf.argmax(), self.nd_pdf.shape)
        normed_grids = Interpd_grids.grids[DF_obs.norm_line + "_norm"]
        grid_fluxes_max = [normed_grids[l][inds_max] for l in DF_best.index]
        DF_best["Model"] = grid_fluxes_max
        
        if NB_Result.deredden: # If we dereddened the observed fluxes at each gridpoint
            obs_flux_dered = [arr[inds_max] for arr in NB_Result.obs_flux_arrs]
            obs_flux_err_dered = [arr[inds_max] for arr in NB_Result.obs_flux_err_arrs]
            DF_best["Obs_dered"] = obs_flux_dered
            DF_best["Flux_err_dered"] = obs_flux_err_dered
            DF_best["Obs_S/N_dered"] = (DF_best["Obs_dered"].values /
                                        DF_best["Flux_err_dered"].values   )
            DF_best["Delta_(SDs)"] = ((DF_best["Model"].values - DF_best["Obs_dered"].values)
                                        / DF_best["Flux_err_dered"].values )
            # Columns to include in output and their order (index is "Line"):
            cols_to_include = ["Model", "Obs_dered", "Obs_S/N_dered", "Delta_(SDs)"]
        else:
            DF_best["Obs_S/N"] = DF_best["Obs"].values / DF_best["Flux_err"].values
            DF_best["Delta_(SDs)"] = ((DF_best["Model"].values - DF_best["Obs"].values)
                                        / DF_best["Flux_err"].values )
            # Columns to include in output and their order (index is "Line"):
            cols_to_include = ["Model", "Obs", "Obs_S/N", "Delta_(SDs)"]
        
        self.best_model["table"] = DF_best[cols_to_include]



    def _calculate_chi2(self, deredden, DF_obs):
        """
        Calculate a chi^2 value which describes how well the model
        corresponding to the parameter best estimates matches the observations.
        Any upper bounds are not included in the calculation.
        deredden: Boolean.  Did we deredden the observed line fluxes to match
                  the Balmer decrement at every interpolated model gridpoint?
        DF_obs: pandas DataFrame holding observed fluxes and errors
        """
        DF = self.best_model["table"]  # Table comparing obs with best model
        grid_n = self.Grid_spec.ndim   # Number of grid dimensions
        # Notes on degrees of freedom calculation: The normalisation line
        # doesn't count as a data point, since its flux is normalised to 1.  If
        # we're dereddening, Halpha also doesn't count because we match the
        # Balmer decrement, so Halpha always matches the prediction.
        # Also, upper bounds aren't included in the chi2 calculation.
        if deredden:  # If we dereddened the observed fluxes at each gridpoint
            # Note that upper bounds can't be used with dereddening.
            # Reconstruct the error column:
            flux_err = DF["Obs_dered"].values / DF["Obs_S/N_dered"].values
            chi2 = np.sum( (DF["Obs_dered"].values - DF["Model"].values)**2
                                         / flux_err**2  )
            dof = (len(DF) - 1 - 1) - grid_n  # Degrees of freedom
        else: # If there wasn't dereddening
            # There may be upper bounds, for which the observed flux is -inf;
            # these can't make a contribution to the chi2.
            flux_err = DF_obs["Flux_err"].values
            chi2, dof = 0.0, 0
            for obs_flux, obs_err, mod_val in zip(DF["Obs"].values, flux_err,
                                                  DF["Model"].values):
                if obs_flux != -np.inf:  # If not an upper bound
                    chi2 += (obs_flux - mod_val)**2 / obs_err**2
                    dof += 1
            dof = (dof - 1) - grid_n  # Degrees of freedom (exclude norm line)

        dof = max(1, dof)  # Can't be smaller than one
        chi2 /= dof  # The "reduced chi-squared"
        self.best_model["chi2"] = chi2



    def _calculate_Av(self, deredden, Interpd_grids, DF_obs):
        """
        Calculate the visual extinction Av in magnitudes that is implied by
        the "best" model.
        deredden: Boolean.  Did we deredden the observed line fluxes to match
                  the Balmer decrement at every interpolated model gridpoint?
        Interpd_grids: Object storing interpolated model emission line grids
        DF_obs: DataFrame containing the input observed line fluxes and errors
        """
        if not deredden:
            self.best_model["extinction_Av_mag"] = "NA (deredden is False)"
            return
        # Find the Balmer decrements for both the "best" model and the raw
        # observations
        inds_max = np.unravel_index(self.nd_pdf.argmax(), self.nd_pdf.shape)
        normed_grids = Interpd_grids.grids[DF_obs.norm_line + "_norm"]
        Ha_Hb_max = [normed_grids[l][inds_max] for l in ["Halpha", "Hbeta"]]
        BD_model = Ha_Hb_max[0] / Ha_Hb_max[1]  # Balmer decrement (predicted)
        BD_obs = DF_obs.loc["Halpha", "Flux"] / DF_obs.loc["Hbeta", "Flux"]

        if BD_model <= BD_obs:  # The expected case
            Av = Av_from_BD(BD_low=BD_model, BD_high=BD_obs)
        else:  # We only deredden where BD_predicted < BD_obs
            Av = 0
        self.best_model["extinction_Av_mag"] = Av



def make_single_parameter_estimate(param_name, val_arr, pdf_1D):
    """
    Bayesian parameter estimate for a single parameter, including the credible
    intervals.  This function is also used to make "estimates" using the prior
    and likelihood PDFs, which may be of interest but this isn't full Bayesian
    parameter estimation.
    param_name: String giving name of the parameter.
    val_arr: 1D numpy array of parameter co-ordinate values associated with
             the values listed in pdf_1D
    pdf_1D: 1D numpy array of the marginalised 1D PDF for the parameter
            param_name; the PDF values correspond to the parameter values
            listed in val_arr.
    Returns a dictionary.  See "_make_parameter_estimate_table" for contents.
    """
    if not np.all(np.isfinite(pdf_1D)):
        raise ValueError("The 1D PDF for "
                         "{0} is not all finite".format(param_name))
    if np.any(pdf_1D < 0):
        raise ValueError("The 1D PDF for "
                         "{0} has a negative value".format(param_name))
    assert val_arr.size == pdf_1D.size  # Sanity check
    index_array = np.arange(val_arr.size)
    bool_map = {True: "Y", False: "N"}
    CIs = [68, 95]  # Credible intervals to consider
    out_dict = {}  # To hold results for this parameter
    out_dict["Parameter"] = param_name

    if np.all(pdf_1D == 0):
        # This is possible if all models are a terrible fit to the data
        out_dict["Estimate"] = val_arr[0]  # Take lowest parameter value
        for CI in CIs:
            CI_code = "CI" + str(CI)  # e.g. "CI68"
            out_dict[CI_code+"_low"] = -np.inf
            out_dict[CI_code+"_high"] = +np.inf
            out_dict["Est_in_"+CI_code+"?"] = bool_map[False]
        other_checks = {
            "n_local_maxima": 0,
            "P(lower)": 0., "P(lower)>50%?": bool_map[False],
            "P(upper)": 0., "P(upper)>50%?": bool_map[False],
            "Est_at_lower?": bool_map[True], "Est_at_upper?": bool_map[True],
            }
        out_dict.update(other_checks)
        return out_dict

    # Calculate estimate of parameter value (location of max in 1D pdf)
    # (This is the Bayesian parameter estimate if pdf_1D is from the posterior)
    est_ind = np.argmax(pdf_1D)
    out_dict["Estimate"] = val_arr[est_ind]
    
    # Generate cumulative density function (CDF) using trapezoidal integration:
    cdf_1D = cumtrapz(pdf_1D, x=val_arr, initial=0)
    # initial=0 => cdf_1D has same length as pdf_1D; first CDF entry will be 0;
    # the last will be 1 (once normalised).
    # Normalise CDF (should be very close to normalised already):
    cdf_1D /= cdf_1D[-1]

    # Calculate credible intervals
    for CI in CIs:
        # Find the percentiles of lower and upper bounds of CI:
        lower_prop = (1.0 - CI/100.0) / 2.0 # Lower percentile as proportion
        upper_prop = 1.0 - lower_prop       # Upper percentile as proportion
        
        # Find value corresponding to the lower bound
        lower_ind_arr = index_array[cdf_1D < lower_prop]
        if lower_ind_arr.size == 1:
            # The 1st CDF value (0) is the only value below the lower bound
            lower_val = -np.inf  # Indicate that we don't have a lower bound
        else: # Use the first value below the lower bound, to be conservative
            lower_val = val_arr[ lower_ind_arr[-1] ]

        # Find value corresponding to the upper bound
        upper_ind_arr = index_array[cdf_1D > upper_prop]
        if upper_ind_arr.size == 1:
            # The last CDF value (1) is the only value above the upper bound
            upper_val = np.inf # Indicate that we don't have an upper bound
        else: # Use the first value above the upper bound, to be conservative
            upper_val = val_arr[ upper_ind_arr[0] ]

        # Save results into output dictionary:
        CI_code = "CI" + str(CI)  # e.g. "CI68"
        out_dict[CI_code+"_low"] = lower_val
        out_dict[CI_code+"_high"] = upper_val

        # Check best estimate is actually inside the CI:
        is_in_CI = (lower_val <= out_dict["Estimate"] <= upper_val)
        out_dict["Est_in_"+CI_code+"?"] = bool_map[is_in_CI]

    # Count local maxima
    tup_max_inds = argrelextrema(pdf_1D, np.greater)
    out_dict["n_local_maxima"] = tup_max_inds[0].size
    # argrelextrema doesn't pick up maxima on the ends, so check ends:
    if pdf_1D[0] > pdf_1D[1]:
        out_dict["n_local_maxima"] += 1
    if pdf_1D[-1] > pdf_1D[-2]:
        out_dict["n_local_maxima"] += 1

    # Check if the mass of probabiltiy is at an edge of the parameter space:
    # Lower bound: Check 3rd value of CDF (the 1st is 0 by design)
    out_dict["P(lower)"] = cdf_1D[2]
    out_dict["P(lower)>50%?"] = bool_map[ (out_dict["P(lower)"] > 0.5) ]
    out_dict["Est_at_lower?"] = bool_map[ (est_ind <= 2) ]
    # Upper bound: Check 3rd-last value of CDF (the last is 1 by design)
    out_dict["P(upper)"] = 1.0 - cdf_1D[-3]
    out_dict["P(upper)>50%?"] = bool_map[ (out_dict["P(upper)"] > 0.5) ]
    out_dict["Est_at_upper?"] = bool_map[ (est_ind >= cdf_1D.size - 4) ]
    
    return out_dict



class NB_Result(object):
    """
    Class to hold the NebulaBayes results including the likelihood, prior and
    posterior, marginalised PDFs and parameter estimates.
    An instance of this class is returned to the user each time NebulaBayes is
    run.
    """
    def __init__(self, Interpd_grids, DF_obs, ND_PDF_Plotter, Plot_Config,
                                          deredden, input_prior, line_plot_dir):
        """
        Initialise an instance of the class and perform Bayesian parameter
        estimation.
        """
        self.DF_obs = DF_obs # Store for user
        self.Plotter = ND_PDF_Plotter # To plot ND PDFs
        self.Plot_Config = Plot_Config
        self.deredden = deredden # Boolean - dereddeden obs fluxes over whole grid?
        self._line_plot_dir = line_plot_dir
        Grid_spec = Grid_description(Interpd_grids.param_names,
                           list(Interpd_grids.paramName2paramValueArr.values()))
        self.Grid_spec = Grid_spec

        # Make arrays of observed fluxes over the grid (possibly dereddening)
        self._make_obs_flux_arrays(Interpd_grids, DF_obs.norm_line)

        # Ensure there are interpolated arrays normalised to the norm_line
        self._normalise_grid_arrays(Interpd_grids, DF_obs.norm_line)

        # Calculate the prior over the grid:
        NB_logger.info("Calculating prior...")
        raw_prior = calculate_prior(input_prior, DF_obs,
                                    Interpd_grids.grids["No_norm"],
                                    grid_spec=Interpd_grids._Grid_spec,
                                    grid_rel_err=Interpd_grids.grid_rel_error)
        self.Prior = NB_nd_pdf(raw_prior, self, Interpd_grids, name="Prior",
                                                                DF_obs=DF_obs)

        # Calculate the likelihood over the grid:
        NB_logger.info("Calculating likelihood...")
        raw_likelihood = self._calculate_likelihood(Interpd_grids,
                                                    DF_obs.norm_line)
        self.Likelihood = NB_nd_pdf(raw_likelihood, self, Interpd_grids,
                                    name="Likelihood", DF_obs=DF_obs)

        # Calculate the posterior using Bayes' Theorem:
        # (note that the prior and likelihood pdfs are now normalised)
        NB_logger.info("Calculating posterior using Bayes' Theorem...")
        raw_posterior = self.Likelihood.nd_pdf * self.Prior.nd_pdf
        if np.all(raw_posterior == 0):
            NB_logger.warning("The posterior is all zero.  The prior and "
                          "likelihood have together completely excluded all "
                          "models in the grid, within the numerical precision")
        self.Posterior = NB_nd_pdf(raw_posterior, self, Interpd_grids,
                                   name="Posterior", DF_obs=DF_obs)



    def _make_obs_flux_arrays(self, Interpd_grids, norm_line):
        """
        Make observed flux arrays covering the entire grid, in preparation for
        calculating the likelihood.
        If requested by the user ("deredden" is True), the observed fluxes are
        dereddened to match the Balmer decrement everywhere in the grid.  The
        observed fluxes are not modified wherever the observed Balmer decrement
        is smaller than the predicted Balmer decrement.
        If "deredden" is False, the observed flux value for a line is uniform
        over the n-D grid.
        The observed fluxes have already been normalised to norm_line.

        Creates the obs_flux_arrs and obs_flux_err_arrs attributes, which hold
        arrays corresponding to the observed linelist in self.DF_obs.
        """
        DF_obs = self.DF_obs
        if not self.deredden:
            # Use the input observed fluxes, which presumably were already
            # dereddened if necessary.  The observed fluxes/errors have already
            # been normalised to the norm_line.
            shape = Interpd_grids.shape
            self.obs_flux_arrs = [np.full(shape, f) for f in
                                                    DF_obs["Flux"].values]
            self.obs_flux_err_arrs = [np.full(shape, e) for e in
                                                    DF_obs["Flux_err"].values]
            # These fluxes/errors remain normalised to the chosen norm_line
            return

        # Deredden observed fluxes at every interpolated gridpoint to match
        # the model Balmer decrement at that gridpoint.
        if norm_line != "Hbeta":
            raise ValueError("Dereddening is only supported for "
                             "norm_line == 'Hbeta'")
        if np.any(DF_obs["Flux"].values == -np.inf):
            raise ValueError("Upper bounds can't be included when dereddening.")
        # Array of Balmer decrements across the grid:
        grid_BD_arr = (Interpd_grids.grids["No_norm"]["Halpha"] /
                       Interpd_grids.grids["No_norm"]["Hbeta"]    )
        if not np.all(np.isfinite(grid_BD_arr)): # Sanity check
            raise ValueError("Something went wrong - the array of predicted"
                             " Balmer decrements over the grid contains a"
                             " non-finite value!")
        if not np.all(grid_BD_arr > 0): # Sanity check
            raise ValueError("Something went wrong - the array of predicted"
                             " Balmer decrements over the grid contains at"
                             " least one non-positive value!")

        obs_flux_arrs, obs_flux_err_arrs = do_dereddening(
                    DF_obs["Wavelength"].values, DF_obs["Flux"].values,
                    DF_obs["Flux_err"].values, BD=grid_BD_arr, normalise=True)
        # The output fluxes and errors are normalised to Hbeta == 1.
        # Now obs_flux_arrs is a list of arrays corresponding to the list
        # of observed fluxes, where each array has the same shape as the 
        # model grid.  The list obs_flux_err_arrs is the same, but for errors.

        # Where the observed Balmer decrement was less than the theoretical
        # Balmer decrement, we undo the dereddening.  Otherwise we're
        # "reddening" the observed spectum for comparison with the models,
        # which is nonsensical!
        obs_BD = DF_obs.loc["Halpha", "Flux"] / DF_obs.loc["Hbeta", "Flux"]
        where_bad_BD = (grid_BD_arr >= obs_BD)
        if np.any(where_bad_BD):
            n_bad = np.sum(where_bad_BD)
            NB_logger.warning("Observed Balmer decrement is below the " +
                        "predicted decrement at {0} ".format(n_bad) +
                        "interpolated gridpoints.  Dereddening will not be " +
                        "applied at these points")
        for flux_arr, obs_flux in zip(obs_flux_arrs, DF_obs["Flux"].values):
            flux_arr[where_bad_BD] = obs_flux
        for err_arr, obs_err in zip(obs_flux_err_arrs, DF_obs["Flux_err"].values):
            err_arr[where_bad_BD] = obs_err
        if not np.allclose(obs_flux_arrs[DF_obs.index.get_loc("Hbeta")], 1.0):
            raise ValueError("Something went wrong - fluxes not normalised")

        for arr, err_arr in zip(obs_flux_arrs, obs_flux_err_arrs):
            if not np.all(np.isfinite(arr)) and np.all(np.isfinite(err_arr)):
                raise ValueError("Something went wrong - the dereddened "
                                 "fluxes and errors are not all finite")

        self.obs_flux_arrs = obs_flux_arrs
        self.obs_flux_err_arrs = obs_flux_err_arrs



    def _normalise_grid_arrays(self, Interpd_grids, norm_line):
        """
        Normalise the interpolated model grid fluxes if necessary (if we don't
        already have a dict of grids with the desired normalisation, from a
        previous NebulaBayes parameter estimation run).
        When we normalise we may lose information (where the normalising grid
        has value zero), so the "No_norm" dict of grids is stored to be able
        to normalise on the fly without this problem.  We store the
        interpolated grids for the last used normalisation to try to avoid
        normalising interpolated grids on every call, which means we store
        two sets of interpolated grids at any time ("No_norm" and the last
        used interpolation).  When we want a new normalisation, we add another
        dict to the "grids" dict and remove the old normalisation.
        """
        norm_name = norm_line + "_norm"
        if norm_name not in Interpd_grids.grids:
            Interpd_grids.grids[norm_name] = OD()  # New dict of grids
            norm_grid = Interpd_grids.grids["No_norm"][norm_line]#.copy()
            # Copy norm_grid so it won't become all "1.0" in the middle of
            # normalising if normalising the same set of grids (we're not)
            bad = (norm_grid == 0)  # For when we divide by norm_grid
            for line, grid in Interpd_grids.grids["No_norm"].items():
                Interpd_grids.grids[norm_name][line] = grid / norm_grid
                # Replace any NaNs we produced by dividing by zero:
                Interpd_grids.grids[norm_name][line][bad] = 0
            if len(Interpd_grids.grids) > 2:
                # Don't store too many copies of the interpolated grids for
                # different normalisations - this might take a lot of memory
                oldest_norm = list(Interpd_grids.grids.keys())[1]
                # The 0th key is "No_norm"; 1st key is for oldest normalisation
                Interpd_grids.grids.popitem(oldest_norm)



    def _calculate_likelihood(self, Interpd_grids, norm_line):
        """
        Calculate the (linear) likelihood over the entire N-D grid at once.
        Returns the likelihood as an nD array.  The likelihood is not yet
        normalised - that will be done later.
        The emission line grids have been interpolated prior to being inputted
        into this method, and are normalised to the norm_line here if this
        hasn't been done already.
        The likelihood is a product of PDFs, one for each contributing emission
        line.  We save out a plot of the PDF for each line if the uses wishes.
        """
        # Systematic relative error in normalised grid fluxes as a linear
        # proportion:
        pred_flux_rel_err = Interpd_grids.grid_rel_error

        # Arrays of observed fluxes over the whole grid (may have been
        # dereddened at every point in the grid)
        obs_flux_arrs = self.obs_flux_arrs
        obs_flux_err_arrs = self.obs_flux_err_arrs

        # We calculate the log of the likelihood, which helps avoid numerical
        # issues in parts of the grid where the models fit the data very badly.
        # Initialise log likelihood with 0 everywhere
        log_likelihood = np.zeros(Interpd_grids.shape, dtype="float")
        for i, line in enumerate(self.DF_obs.index):
            pred_flux_i = Interpd_grids.grids[norm_line + "_norm"][line]
            if np.all(pred_flux_i == 0):
                raise ValueError("Pred flux for {0} all zero".format(line))
            obs_flux_i = obs_flux_arrs[i]
            obs_flux_err_i = obs_flux_err_arrs[i]
            assert obs_flux_i.shape == pred_flux_i.shape
            
            # Calculate the total variance, which is the sum of variances due
            # to both the measured and modelled fluxes:
            # var = obs_flux_err_i**2 + (pred_flux_rel_err * pred_flux_i)**2
            # Rewrite this expression to do in-place manipulation of arrays,
            # which is significantly faster than allocating intermediate arrays
            var_part_1 = obs_flux_err_i.copy()  # Copy - don't modify array
            var_part_1 *= var_part_1  # Squared (in-place array multiplication)
            var_part_2 = pred_flux_i.copy()  # Copy - don't modify array
            var_part_2 *= pred_flux_rel_err
            var_part_2 *= var_part_2  # Squared
            var = var_part_1  # "var" and "var_part_1" refer to the same array
            var += var_part_2  # In-place addition

            if obs_flux_i[tuple(0 for _ in obs_flux_i.shape)] != -np.inf:
                # Minus infinity would signal an upper bound
                # Line contribution with both observed flux and error:
                # log_line_contribution = (-0.5 * np.log(var)
                #     - ((obs_flux_i - pred_flux_i)**2 / (2.0 * var)) )
                # Rewrite this to do in-place manipulation of arrays, as
                # log_cont = - 0.5 * ( (obs - pred)**2 / var + log(var) )
                cont_part_1 = obs_flux_i.copy()  # Copy - don't modify array
                cont_part_1 -= pred_flux_i
                cont_part_1 *= cont_part_1  # Squared
                cont_part_1 /= var
                cont_part_2 = np.log(var)  # Natural log
                log_line_contribution = cont_part_1  # These two names refer to
                                                     # the same array
                log_line_contribution += cont_part_2
                log_line_contribution *= -0.5
            else:  # We have an upper bound
                # Line contribution with only the observed error, not flux:
                denom = np.sqrt(2 * var)
                line_contribution = (erf(pred_flux_i / denom) -
                                  erf((pred_flux_i - obs_flux_err_i) / denom))
                # There can be zeroes in the line_contribution array where
                # pred_flux_i / denom is very large.  We temporarily disable
                # warnings about taking a log of zero here.  The locations
                # with a zero will end up being zero in the final likelihood.
                # There is a risk this will override the information in other
                # lines - but we can't do anything about it.
                np.seterr(divide="ignore")
                log_line_contribution = np.log(line_contribution)  # log base e
                np.seterr(divide="warn")

            log_likelihood += log_line_contribution

            # Plot the ND PDF for each line if requested:
            if self._line_plot_dir is not None:
                log_line_contribution -= log_line_contribution.max()
                line_pdf = np.exp(log_line_contribution)
                outname = os.path.join(self._line_plot_dir,
                                   line + "_PDF_contributes_to_likelihood.pdf")
                Line_PDF = NB_nd_pdf(line_pdf, self, Interpd_grids,
                                     name="Individual_line")  # This name is
                                    # to choose the correct plot config options
                Line_PDF.Grid_spec.param_display_names = \
                                            Interpd_grids.param_display_names
                NB_logger.info("    Plotting PDF for line {0}...".format(line))
                self.Plotter(Line_PDF, outname, config=self.Plot_Config)

        # Roughly normalise; will be properly normalised later
        log_max = log_likelihood.max()  # "max" chooses any number over -inf
        if log_max != -np.inf:  # log_likelihood could be all -inf
            log_likelihood -= log_likelihood.max()

        likelihood = np.exp(log_likelihood)  # The linear likelihood n-D array
        if np.all(likelihood == 0):  # If log_likelihood was all -inf
            NB_logger.warning("The likelihood is all zero - no models are a "
                            "reasonable fit to the data")
        return likelihood


