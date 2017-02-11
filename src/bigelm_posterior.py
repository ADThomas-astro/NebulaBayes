from __future__ import print_function, division
import itertools
from collections import OrderedDict as OD
import numpy as np  # Core numerical library
import pandas as pd
from scipy.integrate import cumtrapz
from scipy.signal import argrelextrema



"""
Adam D. Thomas 2015 - 2017

Functions to calculate the posterior over an N-D grid, marginalise the posterior
to 1D and 2D marginalised posteriors, and do Bayesian parameter estimation.

"""


def calculate_posterior(Grid_container, Obs_Container, log_prior_func):
    """
    Calculate the posterior over the entire N-D grid at once using Bayes'
    Theorem.  The emission line grids are interpolated prior to calculating
    the posterior.
    """
    # Use systematic uncertainty in modelled fluxes, as in Blanc et al.
    epsilon = 0.15 # dex.  Default is 0.15 dex systematic uncertainty
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

    lines_list = Obs_Container.lines_list
    obs_fluxes = Obs_Container.obs_fluxes
    obs_flux_errors = Obs_Container.obs_flux_errors
    grids = Grid_container.Interpd_grids.grids

    # Calculate likelihood:
    # Initialise likelihood with 1 everywhere (multiplictive identity)
    log_likelihood = np.zeros(Grid_container.Interpd_grids.shape, dtype="float")
    for i, emission_line in enumerate(lines_list):
        # Use a log version of equation 3 on pg 4 of Blanc et al. 2015 (IZI)
        # N.B. var is the sum of variances due to both the measured and modelled fluxes
        pred_flux_i = grids[emission_line] # N-D array of predicted fluxes (must be non-negative)
        var = obs_flux_errors[i]**2 + (epsilon_2 * pred_flux_i)**2
        log_likelihood += ( - (( obs_fluxes[i] - pred_flux_i )**2 / 
                                 ( 2.0 * var ))  -  0.5*np.log( var ) )
        # N.B. "log" is base e
    log_likelihood += np.log(2 * np.pi)

    # Note: ??????? The parameter space, space of measurements and space of predictions are all continuous.
    # Each value P in the posterior array is differential, i.e. P*dx = (posterior)*dx
    # for a vector of parameters x.  # ???????????????? ADT - I don't understand my own note

    log_prior = log_prior_func(Grid_container)
    posterior = log_likelihood + log_prior  # Bayes theorem
    posterior -= posterior.max() # "Normalise" so we can return to linear space
    return 10**posterior  # Return linear posterior N-D array



def uniform_prior(Grid_container):
    """
    Return the natural logarithm of a uniform prior.
    Grid_container: Contains details of the input model grid
    Returns an array of the value of a uniform prior over the grid.
    """
    ranges = [(a.max() - a.min()) for a in Grid_container.Interpd_grids.val_arrs]
    prior = np.ones(Grid_container.Interpd_grids.shape, dtype="float")
    prior /=  np.product( ranges )
    return prior




def marginalise_posterior(posterior, Params, val_arrs):
    """
    Calculate normalised 1D and 2D posterior for all possible combinations of
    parameters.
    """
    spacing = [ (v[1] - v[0]) for v in val_arrs ] # In order of params/dimensions
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
                dx=spacing[param_index] )

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
            dx=spacing[1] )

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
                                                     axis=0, dx=spacing[0] )

    #--------------------------------------------------------------------------
    # Calculate the 0D marginalised posterior pdf (by which I mean find
    # the normalisation constant - the 0D marginalised posterior should be 1!)
    # Then normalise the 1D and 2D marginalised posteriors:

    # Firstly find the integral over all Params.n_params dimensions by picking
    # any 1D marginalised posterior (we use the first) and integrating over it:
    integral = np.trapz( marginalised_posteriors_1D[ Params.names[0] ],
                         dx=spacing[0] )
    # print( "Integral for un-normalised full posterior is " + str(integral) )
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

    return marginalised_posteriors_1D, marginalised_posteriors_2D, posterior



#============================================================================
def do_single_parameter_estimate(param_name, val_arr, posterior_1D):
    """
    Bayesian parameter estimate for a single parameter, including the credible
    intervals.
    param_name: String giving name of the parameter.
    val_arr: 1D numpy array of parameter co-ordinate values associated with
             the values listed in posterior_1D
    posterior_1D: 1D numpy array of the marginalised posterior pdf for the
             parameter param_name; the probabilities correspond to the parameter
             values listed in val_arr.
    Returns a dictionary.  See "make_parameter_estimate_table" for contents.
    """

    # Initialise and do some checks:
    length = val_arr.size # Length of val_arr and posterior_1D
    assert length == posterior_1D.size # Sanity check
    
    # Initialise quantities for later use
    index_array = np.arange(length)
    bool_map = {True:"Y", False:"N"}
    out_dict = {}  # To hold results for this parameter
    
    out_dict["Parameter"] = param_name
    # Calculate best estimate of parameter (location of max in posterior):
    est_ind = np.argmax( posterior_1D )
    out_dict["Estimate"] = val_arr[ est_ind ]
    
    # Generate posterior CDF using trapezoidal integration:
    posterior_1D_CDF = cumtrapz( posterior_1D, x=val_arr, initial=0 )
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
    tup_max_inds = argrelextrema( posterior_1D, np.greater )
    out_dict["n_local_maxima"] = tup_max_inds[0].size
    # argrelextrema doesn't pick up maxima on the ends, so check ends:
    if posterior_1D[0] > posterior_1D[1]:
        out_dict["n_local_maxima"] += 1
    if posterior_1D[-1] > posterior_1D[-2]:
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
    # if param_name == "E_peak": raise Exception
    # print(posterior_1D, posterior_1D_CDF)
    return out_dict



#============================================================================
def make_parameter_estimate_table(marginalised_posteriors_1D, val_arrs_dict):
    """
    Do Bayesian parameter estimation for all parameters in the grid.
    marginalised_posteriors_1D: Dicionary mapping parameter names to 1D numpy
                    arrays of marginalised posterior PDFs; the probabilities
                    correspond to the parameter values listed in val_arrs_dict.
    val_arrs_dict:  Mapping of parameter names to 1D numpy arrays of co-ordinate
                    values taken by the parameter in the grid.
    Returns a pandas DataFrame with a row for each parameter.
    """
    print("Calculating parameter estimates...")
    # Initialise and do some checks:
    n_params = len(marginalised_posteriors_1D)
    assert n_params == len(val_arrs_dict)

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
    DF_estimates = pd.DataFrame( OD(
                           [(c, np.zeros(n_params, dtype=t)) for c,t in columns]
                                    )   )
    for col in [col for col,t in columns if t==np.float]:
        DF_estimates.loc[:,col] = np.nan
    DF_estimates.loc[:,"n_local_maxima"] = -1

    for i, (param_name,posteriors_1D) in enumerate(marginalised_posteriors_1D.items()):
        val_arr = val_arrs_dict[param_name]
        p_dict = do_single_parameter_estimate(param_name, val_arr, posteriors_1D)
        for field, value in p_dict.items():
            DF_estimates.loc[i,field] = value

    # Sort DF before returning, so DF is deterministic (note Upper and lower
    # case parame names are sorted into separate sections)
    return DF_estimates.sort_values(by="Parameter")


def make_posterior_peak_table(posterior, Interpd_grids, lines_list, obs_fluxes,
                                                               obs_flux_errors):
    """
    Make a pandas dataframe containing observed emission lines and model fluxes
    for the model corresponding to the peak in the posterior.
    """
    inds_max = np.unravel_index(posterior.argmax(), posterior.shape)
    grid_fluxes_max = [Interpd_grids.grids[line][inds_max] for line in lines_list]
    OD_1 = OD([ ("Line",lines_list), ("Model_flux",grid_fluxes_max),
                ("Obs_flux",obs_fluxes), ("Obs_flux_err",obs_flux_errors)  ])
    DF_peak = pd.DataFrame( OD_1 )
    DF_peak["Obs_S/N"] = DF_peak["Model_flux"] / DF_peak["Obs_flux_err"]
    DF_peak.set_index("Line", inplace=True)
    return DF_peak



def calculate_chi2(DF_peak):
    """
    Calculate chi2 between model and observations for the model corresponding to
    the peak in the posterior.
    """
    chi2_working = ( (DF_peak["Obs_flux"] - DF_peak["Model_flux"])**2
                                 / DF_peak["Obs_flux_err"]**2  )
    chi2 = np.sum(chi2_working.values)
    chi2 /= len(DF_peak) # This is the reduced chi-squared
    return chi2


# def calculate_posterior_stats(posterior, proportions=None):
#     """
#     For each value k in proportions, calculate the fraction of the posterior
#     volume that has a value higher the k times the maximum value of the posterior.
#     proportions: sequence of floats in range [0,1]
#     Returns a list of floats corresponding to the input proportions.
#     """
#     if proportions is None:
#         proportions = np.arange(0.05, 1.02, 0.05)

#     p_max = np.max( posterior )
#     n_posterior = posterior.size

#     results = np.zeros(proportions.size)
#     for k in proportions:
#         results.append( np.sum( posterior > k*p_max ) / n_posterior )

#     return results




