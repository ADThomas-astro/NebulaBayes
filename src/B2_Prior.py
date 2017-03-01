from __future__ import print_function, division
import numpy as np


"""
Adam D. Thomas 2017

Code to determine the function to use to calculate the prior over an N-D grid,
as part of Bayesian parameter estimation.
"""



def calculate_uniform_prior(DF_obs, grids_dict):
    """
    Return the natural logarithm of a uniform prior.
    grids_dict: Dictionary that maps line names to flux arrays
    Returns an array of the value of the prior over the grid.
    """
    shape = list(grids_dict.values())[0].shape
    prior = np.ones(shape, dtype="float")
    # The prior will be normalised later
    return prior



def calculate_He_prior(DF_obs, grids_dict):
    """
    Returns a function that calculates a prior over the N-D interpolated grid,
    based on matching the observed HeII4686/HeI5876 ratio.
    """
    # From simple investigation with S7 data, the HeII4686/HeI5876 ratio appears
    # to vary in over -1.8 < log HeII4686/HeI5876 < 0.9, which is 2.7 dex.
    # I'll use a Gaussian ball with the following std to generate the prior:
    std = 0.5

    obs_ratio = DF_obs.loc["HeII4686", "Flux"] / DF_obs.loc["HeI5876", "Flux"]
    log_obs_ratio = np.log10(obs_ratio)

    # Returns the natural logarithm of the prior over the grid
    grid_ratio = grids_dict["HeII4686"] / grids_dict["HeI5876"]
    # Handle either or both lines having a flux of zero
    bad = (grid_ratio == 0) | (~np.isfinite(grid_ratio))
    grid_ratio[bad] = 1e-250
    log_grid_ratio = np.log10(grid_ratio) # Array with same shape as the grid
    log_prior = - 0.5*((log_grid_ratio - log_obs_ratio) / std)**2
    # The prior will be normalised later
    return log_prior



def calculate_SII_prior(DF_obs, grids_dict):
    """
    Returns a function that calculates a prior over the N-D interpolated grid,
    based on matching the observed SII6731/SII6716 ratio.
    """
    # Now log(SII6731/SII6716) varies between ~-0.17 (density n_e <= 10^2 cm^-3)
    # through ~0.15 (density n_e ~10^3 cm^-3) to ~0.36 (density n_e >= 10^4 cm^-3)
    # This is a range of approximately 0.45 dex.
    # I'll use a Gaussian ball with the following std to generate the prior:
    std = 0.1

    obs_ratio = DF_obs.loc["SII6731", "Flux"] / DF_obs.loc["SII6716", "Flux"]
    log_obs_ratio = np.log10(obs_ratio)

    # Returns the natural logarithm of the prior over the grid
    grid_ratio = grids_dict["SII6731"] / grids_dict["SII6716"]
    # Handle either or both lines having a flux of zero
    bad = (grid_ratio == 0) | (~np.isfinite(grid_ratio))
    grid_ratio[bad] = 1e-250
    log_grid_ratio = np.log10(grid_ratio) # Array with same shape as the grid
    log_prior = - 0.5*((log_grid_ratio - log_obs_ratio) / std)**2
    # The prior will be normalised later
    return log_prior



def calculate_He_and_SII_prior(DF_obs, grids_dict):
    """
    Returns a function that calculates a prior over the N-D interpolated grid,
    based on matching both the observed HeII4686/HeI5876 ratio and the observed
    SII6731/SII6716 ratio.
    """

    log_He_prior = calculate_He_prior(DF_obs, grids_dict)
    log_SII_prior = calculate_SII_prior(DF_obs, grids_dict)
    # The prior will be normalised later
    return log_He_prior + log_SII_prior



def calculate_prior(user_input, DF_obs, grids_dict):
    """
    Calculate the (linear) prior over the grid.
    In the case where the user has not provided a custom function to calculate
    the prior over the grid, we choose the function to calculate the prior based
    on a string.
    user_input: A keyword from the function_map below; which has the default
                value "Uniform"
    DF_obs:     The pandas DataFrame holding the observed fluxes.
    """
    function_map = { "Uniform" : calculate_uniform_prior,
                     "SII_ratio" : calculate_SII_prior,
                     "He_ratio" : calculate_He_prior,
                     "SII_and_He_ratios" : calculate_He_and_SII_prior
                    }
    prior_function = function_map[user_input]
    log_prior = prior_function(DF_obs, grids_dict)
    log_prior -= log_prior.max()
    return np.exp(log_prior)


