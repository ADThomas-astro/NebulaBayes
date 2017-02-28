from __future__ import print_function, division
import numpy as np


"""
Adam D. Thomas 2017

Code to determine the function to use to calculate the prior over an N-D grid,
as part of Bayesian parameter estimation.
"""



def calculate_uniform_prior(Interpd_grids):
    """
    Return the natural logarithm of a uniform prior.
    Interpd_grids: Contains details of the interpolated input model grid
    Returns an array of the value of a uniform prior over the grid.
    """
    ranges = [(h - l) for l, h in Interpd_grids.paramName2paramMinMax.values()]
    prior = np.ones(Interpd_grids.shape, dtype="float")
    prior /=  np.product( ranges )
    return prior



def make_He_prior_func(Interpd_grids, DF_obs):
    """
    Returns a function that calculates a prior over the N-D interpolated grid,
    based on matching the observed HeII4686/HeI5876 ratio.
    """
    # From simple investigation with S7 data, the HeII4686/HeI5876 ratio appears
    # to vary in over -1.8 < log HeII4686/HeI5876 < 0.9, which is 2.7 dex.
    # I'll use a Gaussian with the following std to generate the prior:
    std = 0.5

    obs_ratio = DF_obs.loc["HeII4686", "Flux"] / DF_obs.loc["HeI5876", "Flux"]
    log_obs_ratio = np.log10(obs_ratio)

    def calculate_He_prior(Interpd_grids):
        # Returns the natural logarithm of the prior over the grid
        grid_ratio = Interpd_grids.grids["HeII4686"] / Interpd_grids.grids["HeI5876"]
        # Handle either or both lines having a flux of zero
        bad = (grid_ratio == 0) | (~np.isfinite(grid_ratio))
        grid_ratio[bad] = 1e-250
        log_grid_ratio = np.log10(grid_ratio) # Array with same shape as the grid
        log_prior = - 0.5*((log_grid_ratio - log_obs_ratio) / std)**2
        # Note that the posterior will be normalised later
        return log_prior

    return calculate_He_prior



def make_SII_prior_func(Interpd_grids, DF_obs):
    """
    Returns a function that calculates a prior over the N-D interpolated grid,
    based on matching the observed SII6731/SII6716 ratio.
    """
    # Now log(SII6731/SII6716) varies between ~-0.17 (density n_e <= 10^2 cm^-3)
    # through ~0.15 (density n_e ~10^3 cm^-3) to ~0.36 (density n_e >= 10^4 cm^-3)
    # This is a range of approximately 0.45 dex.
    # I'll use a Gaussian with the following std to generate the prior:
    std = 0.1

    obs_ratio = DF_obs.loc["SII6731", "Flux"] / DF_obs.loc["SII6716", "Flux"]
    log_obs_ratio = np.log10(obs_ratio)

    def calculate_SII_prior(Interpd_grids):
        # Returns the natural logarithm of the prior over the grid
        grid_ratio = Interpd_grids.grids["SII6731"] / Interpd_grids.grids["SII6716"]
        # Handle either or both lines having a flux of zero
        bad = (grid_ratio == 0) | (~np.isfinite(grid_ratio))
        grid_ratio[bad] = 1e-250
        log_grid_ratio = np.log10(grid_ratio) # Array with same shape as the grid
        log_prior = - 0.5*((log_grid_ratio - log_obs_ratio) / std)**2
        # Note that the posterior will be normalised later
        return log_prior

    return calculate_SII_prior


def make_He_and_SII_prior_func(Interpd_grids, DF_obs):
    """
    Returns a function that calculates a prior over the N-D interpolated grid,
    based on matching observed HeII4686/HeI5876 ratio and the observed
    SII6731/SII6716 ratio.
    """
    calculate_He_prior  = make_He_prior_func(Interpd_grids, DF_obs)
    calculate_SII_prior = make_SII_prior_func(Interpd_grids, DF_obs)

    def calculate_He_and_SII_prior(Interpd_grids):
        log_He_prior = calculate_He_prior(Interpd_grids)
        log_SII_prior = calculate_SII_prior(Interpd_grids)
        return log_He_prior + log_SII_prior

    return calculate_He_and_SII_prior





