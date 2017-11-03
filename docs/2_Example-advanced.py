from __future__ import print_function, division
import os
import sys

from astropy.io import fits  # For FITS file I/O
from astropy.table import Table  # Used in converting to pandas DataFrame 
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

# Manipulate path names to load this version of NB, and also to save output
# file in the NebulaBayes/docs subdirectory.
# in the "docs" subdirectory of the NebulaBayes package:
docs_path = os.path.dirname(os.path.realpath(__file__))
NB_parent_dir = os.path.split(os.path.split(docs_path)[0])[0]
sys.path.insert(1, NB_parent_dir)
from NebulaBayes import NB_Model
from NebulaBayes.src.NB2_Prior import calculate_line_ratio_prior



"""
This script shows examples of more advanced usage of NebulaBayes.

There are examples of how to:
 - Filter a model grid table to consider a smaller parameter space than the
   full grid
 - Include an upper bound
 - Include a custom prior
 - Use a custom plotting callback to add annotations to plots
 - Use the built-in dereddening
 - Retrieve information on the parameter estimates and comparison of the
   "best model" from output tables using the NebulaBayes API.


If you move this script from NebulaBayes/docs before running it, you'll need to
change... ?
"""



def filter_NLR_grid():
    """
    Filter the built-in NLR grid to reduce the covered parameter space.
    """
    # First load the binary grid table and convert to a pandas DataFrame table
    grid_table_file = os.path.join(NB_parent_dir, "NebulaBayes", "grids",
                             "NB_NLR_grid.fits.gz")
    BinTableHDU_0 = fits.getdata(grid_table_file, 0)
    DF_grid = Table(BinTableHDU_0).to_pandas()

    # Fix E_peak to the value log10 E_peak/keV = -1.75, reducing the grid from
    # 4 dimensions to 3 (remaining parameters: log U, 12 + log O/H, log P/k)
    DF_grid = DF_grid[DF_grid["log E_peak"] == -1.75]

    # Remove the lowest six oxygen abundances
    abunds = np.unique(DF_grid["12 + log O/H"].values)  # A sorted array
    DF_grid = DF_grid[DF_grid["12 + log O/H"].isin(abunds[6:])]

    return DF_grid



DF_grid = filter_NLR_grid()
grid_params = ["12 + log O/H", "log U", "log P/k"]  # E_peak was removed
interp_shape = [100, 40, 50]  # Interpolated points along each dimension
linelist = ["OII3726_29", "Hbeta", "OIII5007", "OI6300", "Halpha", "NII6583",
            "SII6716", "SII6731"]
obs_fluxes = [1.225,  1,    0.4494, 0.02923, 4.251,  1.653,   0.4560,  0.4148]
obs_errs =  [0.003, 0.0017, 0.0012, 0.00052, 0.0027, 0.00173, 0.00102, 0.00099]
# Must supply wavelengths for dereddening; see NebulaBayes/grids/Linelist.csv
wavelengths = [3727, 4861, 5007, 6300, 6562.8, 6583, 6716.4, 6731]  # Angstroem

# Set up interpolated grids, neglecting any systematic grid error:
NB_Model_1 = NB_Model(DF_grid, grid_params, linelist,
                      interpd_grid_shape=interp_shape, grid_error=0)



def calculate_custom_prior(DF_obs, grids_dict, grid_spec, grid_rel_err):
    """
    Example callback function passed to NebulaBayes to calculate a custom prior.

    Calculate a prior PDF over the N-D interpolated grid, combining a
    contribution from a line-ratio prior with a contribution from information
    on a particular grid parameter (the ionisation parameter). 

    DF_obs:     A pandas DataFrame table holding the observed fluxes/errors
    grids_dict: Dictionary that maps line names to nD interpolated flux arrays.
                These interpolated model flux arrays haven't been normalised
                to the "norm_line" yet.
    grid_spec:  An NB1_Process_grids.Grid_description instance holding basic
                information for the interpolated grids, such as the parameter
                names, parameter values and the grid shape.
    grid_rel_err: The systematic relative error on grid fluxes, as a linear
                  proportion.

    Return an array of the value of the prior over the full interpolated grid.
    """
    # Firstly calculate a contribution to the prior using the NebulaBayes
    # "line-ratio prior" feature, using the density-sensitive ratio of lines in
    # the optical [SII] doublet
    contribution_SII = calculate_line_ratio_prior(DF_obs, grids_dict,
                            grid_rel_err, line_1="SII6716", line_2="SII6731")
    # "contribution_SII" is an array over the full n-D interpolated grid.
    
    # Next calculate a contribution to the prior that varies only with the
    # ionisation parameter, log U.
    # Find the index of the "log U" grid parameter (each parameter corresponds
    # to a grid dimension; this is the index of that grid dimension):
    param_ind = grid_spec.param_names.index("log U")
    # The list of interpolated parameter values for the "log U" grid parameter:
    all_U_values = grid_spec.param_values_arrs[param_ind]  # Sorted 1D array

    # Construct a prior on log U which is uniform in log space below
    # log U = -2.0, and exponentially decreasing (in a half-Gaussian) above
    # this cut
    cut = -2.0  # (log space)
    log_U_1D_prior = np.exp( -((all_U_values - cut) / 0.3)**2 / 2.)
    log_U_1D_prior[log_U_1D_prior <= cut] = 1.0
    # This array has only one dimension.  Construct a slice to use numpy
    # "broadcasting" to apply the 1D prior over the whole 3D grid:
    slice_NLR = [np.newaxis for _ in grid_spec.shape]
    slice_NLR[param_ind] = slice(None)  # "[slice(None)]" means "[:]"
    slice_NLR = tuple(slice_NLR)
    contribution_log_U = np.ones(grid_spec.shape) * log_U_1D_prior[slice_NLR]

    # Combine the two contributions to the prior, weighting them equally
    prior = contribution_SII * contribution_log_U

    # The prior PDF will be properly normalised by NebulaBayes later
    return prior



# def plotting_callback():
#     """

#     """
#     # In a plotting callback, adjust the grid with subplots_adjust:
#     # fig.subplots_adjust(left=grid_bounds["left"], bottom=grid_bounds["bottom"],
#     #         #                                                   wspace=0.04, hspace=0.04)




#     # Close and delete the figure to ensure NB doesn't try to reuse this figure
#     plt.close(fig)
#     del fig




# Configure options for a NB run:
out_path = docs_path
kwargs = {"posterior_plot": os.path.join(out_path,"2_posterior_plot.pdf"),
          "prior_plot": os.path.join(out_path, "2_prior_plot.pdf"),
          "estimate_table": os.path.join(out_path, "2_param_estimates.csv"),
          "prior": calculate_custom_prior,
          "deredden": True,  # Match model Balmer decrement everywhere in grid
          "obs_wavelengths": wavelengths,  # Needed for dereddening
          "norm_line": "Hbeta",  # Obs and model fluxes normalised to Hbeta
          "param_display_names": {"log U": r"$\log U$",
                "log P/k": r"$\log P/k$", "12 + log O/H": r"$12 + \log $O/H"},
          "plot_configs": [{}, {}, {}, {}], # Prior, like, posterior, per-line
          }
# Add "Best model" table to prior plot:
kwargs["plot_configs"][0]["table_on_plot"] = True
# # Use our custom callback function when making the posterior plot:
# kwargs["plot_configs"][2]["callback"] = plotting_callback

# Do parameter estimation once:
Result = NB_Model_1(obs_fluxes, obs_errs, linelist, **kwargs)
# The NB_Model instance can be called repeatedly to do Bayesian parameter
# estimation on different sets of observed fluxes with the same grid (which
# only needs to be interpolated once).
# You can access all the NebulaBayes internal data, PDFs, results, etc. on the
# Result object.



# Now we extract some data:






# Example of use of upper bound
linelist1 = ["OII3726_29", "Hbeta", "OIII5007", "OI6300", "Halpha", "NII6583",
            "SII6716", "SII6731"]
# Include an upper bound on OI6300, signaled by the -inf placeholder:
obs_fluxes1 = [1.225,  1,    0.4494, -np.inf, 4.251,  1.653,   0.4560,  0.4148]
obs_errs1 =  [0.003, 0.0017, 0.0012, 0.04,   0.0027, 0.00173, 0.00102, 0.00099]
kwargs1 = {}  # Use defaults - don't write outputs or deredden
Result1 = NB_Model_1(obs_fluxes1, obs_errs1, linelist1, **kwargs1)



print("Advanced example script complete.")

