from __future__ import print_function, division
import logging
import numpy as np  # Core numerical library
from ._compat import _str_type  # Compatibility


"""
Code to calculate the prior over an n-dimensional grid, as part of Bayesian
parameter estimation.

Adam D. Thomas 2018 - 2020
"""



NB_logger = logging.getLogger("NebulaBayes")



def calculate_uniform_prior(grids_dict):
    """
    Return a (linear probability space) uniform prior.
    grids_dict: Dictionary that maps line names to interpolated n-D predicted
                flux arrays
    Returns an array of the value of the prior over the grid.
    """
    shape = list(grids_dict.values())[0].shape
    prior = np.ones(shape, dtype="float")
    # The prior will be normalised later
    return prior



def calculate_line_ratio_prior(obs_flux_arr_dict, obs_err_arr_dict,
                               grids_dict, grid_rel_err, line_1, line_2):
    """
    Calculate a (linear probability space) prior over the n-dimensional
    interpolated grid by comparing predictions for an emission line ratio with
    the observed ratio.
    obs_flux_arr_dict: A dictionary mapping line names to n-D arrays of
                    observed fluxes over the entire grid.  The fluxes will be
                    the same everywhere unless deredden=True, in which case the
                    fluxes were dereddened to match the predicted Balmer
                    decrement at each point in the interpolated parameter space.
                    The fluxes have been normalised to the input norm_line.
    obs_err_arr_dict:  Same as obs_flux_arr_dict, but for the flux errors.
    grids_dict: Dictionary that maps line names to interpolated n-D predicted
                flux arrays.  These interpolated arrays are based on the input
                model grid and have been normalised to the input norm_line.
    grid_rel_err: Relative systematic error on grid fluxes, as a linear
                  proportion.  This is an input into NebulaBayes.
    line_1 and line_2: Names of emission lines.
    Returns an array of the value of the prior over the grid.  The prior is
    calculated by comparing observed and model values of F(line_1) / F(line_2).
    """
    for line_i in [line_1, line_2]:
        if line_i not in obs_flux_arr_dict:
            raise ValueError("The line {0} was used in a prior but is not in"
                             " the list of observed lines".format(line_i))

    # Calculate arrays of the observed line flux ratio with errors over the
    # full parameter space
    flux_1, err_1 = obs_flux_arr_dict[line_1], obs_err_arr_dict[line_1]
    flux_2, err_2 = obs_flux_arr_dict[line_2], obs_err_arr_dict[line_2]
    obs_ratio = flux_1 / flux_2
    obs_ratio_err = obs_ratio * np.hypot(err_1/flux_1, err_2/flux_2)

    # Find the predicted line ratio, as an n-D array over the interpolated grid
    grid_ratio = grids_dict[line_1] / grids_dict[line_2]
    # Handle either or both lines having a flux of zero
    bad = (grid_ratio == 0) | (~np.isfinite(grid_ratio))
    grid_ratio[bad] = 1e-250

    # Calculate the probability over the n-D grid in the same manner as for
    # the likelihood, comparing the observations to predictions across the grid
    var = obs_ratio_err**2 + (grid_rel_err * grid_ratio)**2
    # Note that the grid_rel_error hasn't been properly propagated here, since
    # it is intended to apply directly to the fluxes, not to a flux ratio.
    # However, it's an approximate way to treat systematic errors anyway.
    log_prior = - ((obs_ratio - grid_ratio)**2 / (2.0 * var)) - 0.5*np.log(var)
    # N.B. "log" is base e
    log_prior -= log_prior.max() # Normalise-ish
    prior = np.exp(log_prior)
    # The prior PDF will be properly normalised later
    return prior



def calculate_prior(user_input, DF_obs, obs_flux_arr_dict, obs_err_arr_dict,
                                          grids_dict, grid_spec, grid_rel_err):
    """
    Calculate the (linear probability space) prior over the grid, selecting the
    type of prior based on the request of the user (or the default).
    In the case where the user has not provided a custom function to calculate
    the prior over the grid or provided the prior as an array, we use either a
    uniform prior (the default), or calculate a prior based on one or more line
    ratios.  We check that the prior is finite and non-negative.
    The user's input specifying how to calculate the prior:
        user_input: Either the string "Uniform", a list of tuples such as
                    [("HeI5876","HeII4686"), ("SII6716","SII6731")] (specifying
                    line ratios), a python "callable" (custom user function),
                    or a numpy ndarray.
    Arguments for the observed data:
        DF_obs:     The pandas DataFrame table holding the observed fluxes,
                    with a row for each line.
        obs_flux_arr_dict: A dictionary mapping line names to n-D arrays of
                    observed fluxes over the entire grid.  The fluxes will be
                    the same everywhere unless deredden=True, in which case the
                    fluxes were dereddened to match the predicted Balmer
                    decrement at each point in the interpolated parameter space.
                    The fluxes have been normalised to the input norm_line.
        obs_err_arr_dict:  Same as obs_flux_arr_dict, but for the flux errors.
    Arguments for the model data:
        grids_dict: Dictionary that maps line names to n-D interpolated flux
                    arrays.  These interpolated arrays are based on the input
                    grid and have been normalised to the input norm_line.
        grid_spec:  A NB1_Process_grids.Grid_description instance holding basic
                    information for the interpolated grids, such as the
                    parameter names, parameter values and the grid shape.
        grid_rel_err: The systematic relative error on grid fluxes, as a linear
                      proportion.  This is an input into NebulaBayes.
    Returns the prior as a numpy ndarray, in linear space.
    """
    if callable(user_input):
        prior = user_input(DF_obs, obs_flux_arr_dict, obs_err_arr_dict,
                           grids_dict, grid_spec, grid_rel_err)
    elif isinstance(user_input, np.ndarray):
        prior = user_input
        if prior.shape != grid_spec.shape:
            raise ValueError("The prior array must have the same shape as the "
                             "interpolated grid")
    elif isinstance(user_input, _str_type):
        if user_input.upper() == "UNIFORM":  # Case-insensitive
            prior = calculate_uniform_prior(grids_dict)
        else:
            raise ValueError("The only string accepted for the 'prior' keyword"
                             " is 'Uniform'")
    elif isinstance(user_input, list):
        assert len(user_input) > 0
        if not all((isinstance(t, tuple) and len(t) == 2) for t in user_input):
            raise ValueError("The list provided for 'prior' must consist only"
                             " of tuples of length 2")
        kwargs = dict(obs_flux_arr_dict=obs_flux_arr_dict,
                      obs_err_arr_dict=obs_err_arr_dict,
                      grids_dict=grids_dict, grid_rel_err=grid_rel_err)
        priors = [calculate_line_ratio_prior(line_1=t0, line_2=t1, **kwargs)
                                                       for t0,t1 in user_input]
        prior = np.product(priors, axis=0)
    else:
        raise ValueError("The input 'prior' must be one of: 'Uniform', a list"
                         " of tuples, a callable, or a numpy array")

    # Check that the prior is sensible:
    if not prior.shape == grid_spec.shape:
        raise ValueError("The prior shape {0} differs from the interpolated "
                         "grid shape {1}".format(prior.shape, grid_spec.shape))
    if not np.all(np.isfinite(prior)):
        raise ValueError("The prior is not entirely finite")
    if prior.min() < 0:
        raise ValueError("The prior contains a negative value")
    if np.all(prior == 0):
        NB_logger.warning("WARNING: The prior is entirely zero")

    # Return linear prior, which will be normalised later
    return prior


