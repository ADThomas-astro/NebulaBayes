from __future__ import print_function, division
import itertools
from collections import OrderedDict as OD
import numpy as np  # Core numerical library
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.signal import argrelextrema
from .dereddening import deredden as do_dereddening
from .B1_Grid_working import Grid_description
from .B2_Prior import calculate_prior



"""
Adam D. Thomas 2015 - 2017

Code to calculate the likelihood and posterior over an N-D grid, marginalise
pdfs to 1D and 2D marginalised pdfs, and generally do Bayesian parameter
estimation.
This module defines two custom NebulaBayes classes: NB_nd_pdf and NB_Result.
"""



class NB_nd_pdf(object):
    """
    Class to hold an N-dimensional PDF and its 1D and 2D marginalised forms.
    An instance of this class is used for each of the likelihood, prior and
    posterior.
    """
    def __init__(self, nd_pdf, NB_Result, DF_obs, Interpd_grids):
        """
        Initialise an instance of the NB_nd_pdf class.
        nd_pdf: A numpy ndarray holding the (linear) pdf.
        NB_Result: A NB_Result object (defined in this module)
        DF_obs: A pandas DataFrame table holding the observed emission line
                fluxes and errors
        Interpd_grids: A NB_Grid object (defined in NB1_Grid_working.py)
                       holding the interpolated model grid fluxes and the
                       description fo the model grid.
        """
        self.nd_pdf = nd_pdf
        self.Grid_spec = NB_Result.Grid_spec
        # Add self.marginalised_2d and self.marginalised_1d attributes and
        # normalise the self.nd_pdf attribute:
        self.marginalise_pdf()
        # Make a parameter estimate table based on this nd_pdf
        self.make_parameter_estimate_table() # add self.DF_estimates attribute
        # Make a table comparing model and observed fluxes at the pdf peak
        self.make_pdf_peak_table(DF_obs, Interpd_grids, NB_Result)
        # We added the self.DF_peak attribute

        # Calculate chi2 at pdf peak (add self.chi2 attribute)
        self.calculate_chi2(NB_Result.deredden)



    def marginalise_pdf(self):
        """
        Calculate normalised 1D and 2D marginalised pdfs for all possible
        combinations of parameters.
        """
        # The interpolated grids have uniform spacing:
        spacing = [(v[1] - v[0]) for v in self.Grid_spec.param_values_arrs]
        n = self.Grid_spec.ndim
        #--------------------------------------------------------------------------
        # Calculate the 2D marginalised pdf for every possible combination
        # of 2 parameters
        print("Calculating 2D marginalised posteriors...")
        # List of all possible pairs of two parameter names:
        param_names = self.Grid_spec.param_names
        double_names = list(itertools.combinations(param_names, 2))
        self.Grid_spec.double_names = double_names
        # Looks something like: [('BBBP','Gamma'), ('BBBP','NT_frac'),
        #                        ('BBBP','UH_at_r_inner'), ('Gamma','NT_frac'),
        #                        ('Gamma','UH_at_r_inner'), ('NT_frac','UH_at_r_inner')]
        # Corresponding list of possible combinations of two parameter indices:
        double_indices = list(itertools.combinations(np.arange(n), 2))
        self.Grid_spec.double_indices = double_indices
        # Looks something like: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        # Note that the order of indices in each tuple is from samller to larger
        
        # Initialise dictionary of all possible 2D marginalised pdf arrays:
        marginalised_2D = {}  # The dict keys will be tuples of 2 parameter names.
        # Iterate over all possible pairs of parameters:
        for double_name, param_inds_double in zip(double_names, double_indices):
            # Generate list of indices/dimensions/parameters to integrate over:
            inds_for_integration = np.arange(n).tolist()  # Initialise
            inds_for_integration.remove( param_inds_double[0] )
            inds_for_integration.remove( param_inds_double[1] )
            inds_for_integration.reverse() # Ensure we integrate over higher dimensions first,
            # so dimension index numbers are still valid after each integration.

            marginalised_2D[double_name] = self.nd_pdf.copy()  # Initialise
            # Keep integrating one dimension at a time until the result only has 2 dimensions:
            for param_index in inds_for_integration:
                # Integrate over this dimension (parameter), using the trapezoidal rule
                marginalised_2D[double_name] = np.trapz( 
                    marginalised_2D[double_name], axis=param_index,
                    dx=spacing[param_index] )

        #--------------------------------------------------------------------------
        # Calculate the 1D marginalised pdf for each individual parameter
        print("Calculating 1D marginalised posteriors...")
        # Initialise dictionary of all 1D marginalised pdf arrays:
        marginalised_1D = {}

        # For the first parameter in param_names:
        # Integrate the first 2D marginalised pdf over the other
        # dimension (parameter), using the trapezoidal rule:
        marginalised_1D[param_names[0]] = np.trapz( 
             marginalised_2D[double_names[0]], axis=1, dx=spacing[1])

        # For all parameters after the first in param_names:
        for double_name, param_inds_double in zip(double_names[:n-1], double_indices[:n-1]):
            # For each pair of parameters we take the second parameter, and integrate 
            # over the first parameter of the pair (which by construction is always the
            # first parameter in param_names).
            assert( param_inds_double[0] == 0 )
            param = param_names[ param_inds_double[1] ]
            # Integrate over first dimension (parameter) using trapezoidal method:
            marginalised_1D[param] = np.trapz(
                 marginalised_2D[double_name], axis=0, dx=spacing[0])

        #--------------------------------------------------------------------------
        # Calculate the 0D marginalised pdf (by which I mean find
        # the normalisation constant - the 0D marginalised pdf should be 1!)
        # Then normalise the 1D and 2D marginalised posteriors:

        # Firstly find the integral over all n dimensions by picking
        # any 1D marginalised pdf (we use the first) and integrating over it:
        integral = np.trapz( marginalised_1D[ param_names[0] ],
                             dx=spacing[0] )
        # print( "Integral for un-normalised full pwd is " + str(integral) )
        # Now actually normalise each 2D and 1D marginalised pdf:
        for double_name in double_names:
            # Divide arrays in-place in memory:
            marginalised_2D[double_name] /= integral
        for param in param_names:
            # Divide arrays in-place in memory:
            marginalised_1D[param] /= integral
        # Now normalise the full pdf.  In the case of the likelihood and prior,
        # this normalisation is important; for the posterior it's probably not.
        self.nd_pdf /= integral

        self.marginalised_2D = marginalised_2D
        self.marginalised_1D = marginalised_1D



    def make_parameter_estimate_table(self):
        """
        Do Bayesian parameter estimation for all parameters in the grid.
        posteriors_marginalised_1D: Dicionary mapping parameter names to 1D numpy
                        arrays of marginalised posterior PDFs; the probabilities
                        correspond to the parameter values listed in val_arrs_dict.
        val_arrs_dict:  Mapping of parameter names to 1D numpy arrays of co-ordinate
                        values taken by the parameter in the grid.
        Creates a pandas DataFrame with a row for each parameter.
        """
        print("Calculating parameter estimates...")
        # Initialise DataFrame to hold results
        # DataFrame columns and types:
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
        DF_estimates = pd.DataFrame(OD1)
        DF_estimates.loc[:,"Parameter"] = self.Grid_spec.param_names
        DF_estimates.sort_values(by="Parameter", inplace=True)
        # Sort DF, so the DF is deterministic (note Upper and lower case param
        # names are sorted into separate sections)
        DF_estimates.set_index("Parameter", inplace=True)
        for col in [col for col,t in columns if t==np.float]:
            DF_estimates.loc[:,col] = np.nan
        DF_estimates.loc[:,"n_local_maxima"] = -1
        
        for p, pdf_1D in self.marginalised_1D.items():
            param_val_arr = self.Grid_spec.paramName2paramValueArr[p]
            p_dict = make_single_parameter_estimate(p, param_val_arr, pdf_1D)
            for field, value in p_dict.items():
                if field != "Parameter":
                    DF_estimates.loc[p,field] = value

        self.DF_estimates = DF_estimates



    def make_pdf_peak_table(self, DF_obs, Interpd_grids, NB_Result):
        """
        Make a pandas dataframe comparing observed emission lines and model
        fluxes for the model corresponding to the peak in the nD PDF.
        """
        DF_peak = DF_obs.copy() # Index: "Line"; columns: "Flux", "Flux_err"
        # DF_obs may also possibly have a "Wavelength" column
        DF_peak.rename(columns={"Flux":"Obs"}, inplace=True)

        inds_max = np.unravel_index(self.nd_pdf.argmax(), self.nd_pdf.shape)
        grid_fluxes_max = [Interpd_grids.grids[l][inds_max] for l in DF_peak.index]
        DF_peak["Model"] = grid_fluxes_max
        
        if NB_Result.deredden: # If we dereddened the observed fluxes at each gridpoint
            obs_flux_dered = [arr[inds_max] for arr in NB_Result.obs_flux_arrs]
            obs_flux_err_dered = [arr[inds_max] for arr in NB_Result.obs_flux_err_arrs]
            DF_peak["Obs_dered"] = obs_flux_dered
            DF_peak["Flux_err_dered"] = obs_flux_err_dered
            DF_peak["Obs_S/N_dered"] = (DF_peak["Obs_dered"].values /
                                        DF_peak["Flux_err_dered"].values   )
            DF_peak["Delta_(SDs)"] = ((DF_peak["Model"].values - DF_peak["Obs_dered"].values)
                                        / DF_peak["Flux_err_dered"].values )
            # Columns to include in output and their order (index is "Line"):
            cols_to_include = ["Model", "Obs_dered", "Obs_S/N_dered", "Delta_(SDs)"]
        else:
            DF_peak["Obs_S/N"] = DF_peak["Obs"].values / DF_peak["Flux_err"].values
            DF_peak["Delta_(SDs)"] = ((DF_peak["Model"].values - DF_peak["Obs"].values)
                                        / DF_peak["Flux_err"].values )
            # Columns to include in output and their order (index is "Line"):
            cols_to_include = ["Model", "Obs", "Obs_S/N", "Delta_(SDs)"]
        
        self.DF_peak = DF_peak[cols_to_include]



    def calculate_chi2(self, deredden):
        """
        Calculate chi2 between model and observations for the model
        corresponding to the peak in the posterior.
        deredden: Boolean.  Did we deredden the observed line fluxes to match
                  the Balmer decrement at every interpolated model gridpoint?
        """
        DF = self.DF_peak # Only contains a subset of useful columns
        # We'll need to reconstruct the error column
        if deredden: # If we dereddened the observed fluxes at each gridpoint
            flux_err = DF["Obs_dered"].values / DF["Obs_S/N_dered"].values
            chi2_working = ( (DF["Obs_dered"].values - DF["Model"].values)**2
                                         / flux_err**2  )
        else:
            flux_err = DF["Obs"].values / DF["Obs_S/N"].values
            chi2_working = ( (DF["Obs"].values - DF["Model"].values)**2
                                         / flux_err**2  )
        chi2 = np.sum(chi2_working)
        grid_n = self.Grid_spec.ndim
        # Degrees of freedom (Hbeta doesn't count as a data point, since it's
        # normalised to 1)
        dof = (len(DF) - 1) - grid_n
        if deredden:
            dof -= 1 # Halpha doesn't count as a data point; the obs fluxes are
                     # dereddened to match the Balmer decrement at every gridpoint
        chi2 /= dof # This is the reduced chi-squared
        self.chi2 = chi2



def make_single_parameter_estimate(param_name, val_arr, pdf_1D):
    """
    Bayesian parameter estimate for a single parameter, including the credible
    intervals.  This function is also used to make "estimates" using the prior
    and likelihood pdfs, which may be of interest but obviously this isn't full
    Bayesian parameter estimation.
    param_name: String giving name of the parameter.
    val_arr: 1D numpy array of parameter co-ordinate values associated with
             the values listed in pdf_1D
    pdf_1D: 1D numpy array of the marginalised posterior pdf for the
             parameter param_name; the probabilities correspond to the parameter
             values listed in val_arr.
    Returns a dictionary.  See "make_parameter_estimate_table" for contents.
    """
    # Initialise and do some checks:
    length = val_arr.size # Length of val_arr and pdf_1D
    assert length == pdf_1D.size # Sanity check
    index_array = np.arange(length)
    bool_map = {True:"Y", False:"N"}
    out_dict = {}  # To hold results for this parameter
    
    out_dict["Parameter"] = param_name
    # Calculate best estimate of parameter (location of max in posterior):
    est_ind = np.argmax( pdf_1D )
    out_dict["Estimate"] = val_arr[ est_ind ]
    
    # Generate posterior CDF using trapezoidal integration:
    posterior_1D_CDF = cumtrapz( pdf_1D, x=val_arr, initial=0 )
    # initial=0 => CDF has same length as PDF; first CDF entry will be 0,
    # last will be 1 (once normalised).  So we've assumed that the
    # posterior PDF probability value is at the RIGHT of each bin.
    # Normalise CDF (should be very close to normalised already):
    posterior_1D_CDF /= posterior_1D_CDF[-1]

    # Calculate credible intervals
    for CI in [68, 95]: # Iterate over credible intervals
        # Find the percentiles of lower and upper bounds of CI:
        lower_prop = (1.0 - CI/100.0) / 2.0 # Lower percentile as proportion
        upper_prop = 1.0 - lower_prop       # Upper percentile as proportion
        
        # Find value corresponding to the lower bound
        lower_ind_arr = index_array[posterior_1D_CDF < lower_prop]
        if lower_ind_arr.size == 1:
            # The 1st CDF value (0) is the only value below the lower bound
            lower_val = -np.inf # Indicate that we don't have a lower bound
        else: # Use the first value below the lower bound, to be conservative
            lower_val = val_arr[ lower_ind_arr[-1] ]

        # Find value corresponding to the upper bound
        upper_ind_arr = index_array[posterior_1D_CDF > upper_prop]
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
    tup_max_inds = argrelextrema( pdf_1D, np.greater )
    out_dict["n_local_maxima"] = tup_max_inds[0].size
    # argrelextrema doesn't pick up maxima on the ends, so check ends:
    if pdf_1D[0] > pdf_1D[1]:
        out_dict["n_local_maxima"] += 1
    if pdf_1D[-1] > pdf_1D[-2]:
        out_dict["n_local_maxima"] += 1

    # Now check the ends to see if the posterior is up against the bounds:
    # Lower bound: Check 3rd value of CDF (the 1st is 0 by design)
    out_dict["P(lower)"] = posterior_1D_CDF[2]
    out_dict["P(lower)>50%?"] = bool_map[ (out_dict["P(lower)"] > 0.5) ]
    out_dict["Est_at_lower?"] = bool_map[ (est_ind <= 2) ]
    # Upper bound: Check 3rd-last value of CDF (the last is 1 by design)
    out_dict["P(upper)"] = 1.0 - posterior_1D_CDF[-3]
    out_dict["P(upper)>50%?"] = bool_map[ (out_dict["P(upper)"] > 0.5) ]
    out_dict["Est_at_upper?"] = bool_map[ (est_ind >= posterior_1D_CDF.size - 4) ]
    return out_dict



class NB_Result(object):
    """
    Class to hold the NebulaBayes results including the likelihood, prior and
    posterior, marginalised posteriors and parameter estimates.
    """
    def __init__(self, Interpd_grids, DF_obs, deredden, input_prior):
        """
        Initialise an instance of the class and perform Bayesian parameter
        estimation.
        """
        self.deredden = deredden # Boolean
        self.ndim = Interpd_grids.ndim
        Grid_spec = Grid_description(Interpd_grids.param_names,
                           list(Interpd_grids.paramName2paramValueArr.values()))
        self.Grid_spec = Grid_spec

        # Calculate arrays of observed fluxes over the grid (possibly dereddening)
        self.make_obs_flux_arrays(DF_obs, Interpd_grids)

        # Calculate the likelihood over the grid:
        raw_likelihood = self.calculate_likelihood(Interpd_grids, DF_obs)
        self.Likelihood = NB_nd_pdf(raw_likelihood, self, DF_obs,
                                                                  Interpd_grids)

        # Calculate the prior over the grid:
        raw_prior = calculate_prior(input_prior, DF_obs, Interpd_grids.grids,
                                                   Interpd_grids.grid_rel_error)
        self.Prior = NB_nd_pdf(raw_prior, self, DF_obs, Interpd_grids)

        # Calculate the posterior using Bayes' Theorem:
        # (note that the prior and likelihood pdfs are now normalised)
        raw_posterior = self.Likelihood.nd_pdf * self.Prior.nd_pdf
        self.Posterior = NB_nd_pdf(raw_posterior, self, DF_obs, Interpd_grids)



    def make_obs_flux_arrays(self, DF_obs, Interpd_grids):
        """
        Make observed flux arrays covering the entire grid.
        If requested by the user, the observed fluxes are dereddened to match
        the Balmer decrement everywhere in the grid.
        """
        if self.deredden:
            # Deredden observed fluxes at every interpolated gridpoint to match
            # the model Balmer decrement at that gridpoint.
            # Array of Balmer decrements across the grid:
            grid_BD_arr = Interpd_grids.grids["Halpha"] / Interpd_grids.grids["Hbeta"]
            # Fluxes should be normalised to Hbeta == 1, but I did the division
            # here explicitly just in case...
            obs_flux_arrs, obs_flux_err_arrs = do_dereddening(
                        DF_obs["Wavelength"].values, DF_obs["Flux"].values,
                        DF_obs["Flux_err"].values, BD=grid_BD_arr, normalise=True)
            # The output fluxes and errors are normalised to Hbeta == 1.
        else: # Use the input observed fluxes, which hopefully have already
              # been dereddened if necessary.
            shape = Interpd_grids.shape
            obs_flux_arrs = [np.full(shape, f) for f in DF_obs["Flux"].values]
            obs_flux_err_arrs = [np.full(shape, e) for e in DF_obs["Flux_err"].values]

        # Now obs_flux_arrs is a list of arrays corresponding to the list
        # of observed fluxes, where each array has the same shape as the 
        # model grid.  The list obs_flux_err_arrs is the same, but for errors.
        self.obs_flux_arrs = obs_flux_arrs
        self.obs_flux_err_arrs = obs_flux_err_arrs



    def calculate_likelihood(self, Interpd_grids, DF_obs):
        """
        Calculate the (linear) likelihood over the entire N-D grid at once.
        The emission line grids are interpolated prior to calculating
        the likelihood.
        """
        # # Use systematic uncertainty in modelled fluxes, as in Blanc et al.
        # epsilon = 0.15 # dex.  Default is 0.15 dex systematic uncertainty
        # # Convert from dex to a scaling factor:
        # epsilon_2 = 10**epsilon - 1  # This gives a factor of 0.41 for epsilon=0.15 dex
        # # epsilon_2 is a scaling factor to go from a linear modelled flux to an
        # # associated systematic error
        # # Note the original from izi.pro is equivalent to:
        # # epsilon_2 = epsilon * np.log(10)
        # # This gives 0.345 for epsilon=0.15 dex. I don't understand this.
        # # Why was a log used?  And a log base e?  This is surely wrong.
        # # What is intended by this formula anyway?
        # # Note that epsilon_2 is the assumed fractional systematic error in the model
        # # fluxes already normalised to Hbeta.  In izi.pro the default is 0.15 dex,
        # # but the default is given as 0.1 dex in the Blanc+15 paper.
        pred_flux_rel_err = Interpd_grids.grid_rel_error

        obs_flux_arrs = self.obs_flux_arrs
        obs_flux_err_arrs = self.obs_flux_err_arrs

        # Initialise likelihood with 1 everywhere (multiplictive identity)
        log_likelihood = np.zeros(Interpd_grids.shape, dtype="float")
        for i, line in enumerate(DF_obs.index): # Iterate over emission lines
            # Use a log version of equation 3 on pg 4 of Blanc et al. 2015 (IZI)
            # N.B. var is the sum of variances due to both the measured and modelled fluxes
            # N-D array of predicted fluxes (must be non-negative):
            pred_flux_i = Interpd_grids.grids[line]
            obs_flux_i = obs_flux_arrs[i]
            obs_flux_err_i = obs_flux_err_arrs[i]
            assert obs_flux_i.shape == pred_flux_i.shape
            var = obs_flux_err_i**2 + (pred_flux_rel_err * pred_flux_i)**2
            log_likelihood += ( - (( obs_flux_i - pred_flux_i )**2 / 
                                     ( 2.0 * var ))  -  0.5*np.log( var ) )
            # N.B. "log" is base e
        # log_likelihood += np.log(2 * np.pi)

        # Note: ??????? The parameter space, space of measurements and space of predictions are all continuous.
        # Each value P in the likelihood array is differential, i.e. P*dx = (likelihood)*dx
        # for a vector of parameters x.  # ???????????????? ADT - I don't understand the purpose of my own note

        log_likelihood -= log_likelihood.max()
        return np.exp(log_likelihood) # The linear likelihood N-D array   


