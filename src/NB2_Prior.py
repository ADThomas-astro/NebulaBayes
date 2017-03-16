from __future__ import print_function, division
import numpy as np # Core numerical library


"""
Adam D. Thomas 2017

Code to calculate the prior over an N-D grid, as part of Bayesian parameter
estimation.
"""



def calculate_uniform_prior(grids_dict):
    """
    Return the natural logarithm of a uniform prior.
    grids_dict: Dictionary that maps line names to flux arrays
    Returns an array of the value of the prior over the grid.
    """
    shape = list(grids_dict.values())[0].shape
    prior = np.ones(shape, dtype="float")
    # The prior will be normalised later
    return prior



def calculate_line_ratio_prior(DF_obs, grids_dict, grid_rel_err, line_1, line_2):
    """
    Returns a (linear) prior calculated over the N-D interpolated grid, based
    on matching an observed emission line ratio.
    DF_obs: pandas Dataframe holding observed line fluxes and errors
    grids_dict: Dictionary that maps line names to flux arrays
    grid_rel_err: Relative systematic error on grid fluxes, as a linear proportion.
    line_1 and line_2: Names of emission lines (index values in DateFrame)
    """
    flux_1, err_1 = DF_obs.loc[line_1, "Flux"], DF_obs.loc[line_1, "Flux_err"]
    flux_2, err_2 = DF_obs.loc[line_2, "Flux"], DF_obs.loc[line_2, "Flux_err"]
    obs_ratio = flux_1 / flux_2
    obs_ratio_err = obs_ratio * np.sqrt( (err_1/flux_1)**2 + (err_2/flux_2)**2 )

    # Returns the natural logarithm of the prior over the grid
    grid_ratio = grids_dict[line_1] / grids_dict[line_2]
    # Handle either or both lines having a flux of zero
    bad = (grid_ratio == 0) | (~np.isfinite(grid_ratio))
    grid_ratio[bad] = 1e-250

    var = obs_ratio_err**2 + (grid_rel_err * grid_ratio)**2
    log_prior = - ((obs_ratio - grid_ratio)**2 / (2.0 * var)) - 0.5*np.log(var)
    # N.B. "log" is base e

    log_prior -= log_prior.max()
    prior = np.exp(log_prior)
    # The prior will be normalised later
    return prior



def calculate_prior(user_input, DF_obs, grids_dict, grid_rel_err):
    """
    Calculate the (linear) prior over the grid, selecting the type of prior
    based on the request of the user (or the default).
    In the case where the user has not provided a custom function to calculate
    the prior over the grid, we choose the function to calculate the prior based
    on a string.
    user_input: A keyword from the function_map below; which has the default
                value "Uniform", or a python "callable".
    DF_obs:     The pandas DataFrame holding the observed fluxes.
    grids_dict: Dictionary that maps line names to flux arrays
    grid_rel_err: The systematic relative error on grid fluxes, as a linear
                  proportion.
    """
    if callable(user_input):
        prior = user_input(DF_obs, grids_dict, grid_rel_err)
    elif user_input == "Uniform":
        prior = calculate_uniform_prior(grids_dict)
    elif user_input == "SII_ratio":
        prior = calculate_line_ratio_prior(DF_obs, grids_dict, grid_rel_err, 
                                                           "SII6731", "SII6716")
    elif user_input == "He_ratio":
        prior = calculate_line_ratio_prior(DF_obs, grids_dict, grid_rel_err, 
                                                           "HeII4686", "HeI5876")
    elif user_input == "SII_and_He_ratios":
        SII_prior = calculate_line_ratio_prior(DF_obs, grids_dict, grid_rel_err, 
                                                           "SII6731", "SII6716")
        He_prior = calculate_line_ratio_prior(DF_obs, grids_dict, grid_rel_err, 
                                                           "HeII4686", "HeI5876")
        prior = SII_prior * He_prior
    else:
        raise ValueError("The input 'prior' must be one of the permitted "
                         "strings or a callable")
    # Return linear prior, which will be normalised later
    return prior


