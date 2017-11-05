from __future__ import print_function, division
import os
import sys
from astropy.io import fits
from astropy.table import Table  # Used in converting to pandas DataFrame 
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd


# Manipulate path names to load the correct version of NB, and also to save the
# output files in the NebulaBayes/docs subdirectory.
DOCS_PATH = os.path.dirname(os.path.realpath(__file__))
NB_PARENT_DIR = os.path.split(DOCS_PATH)[0]
sys.path.insert(1, NB_PARENT_DIR)

from NebulaBayes import NB_Model
from NebulaBayes.NB2_Prior import calculate_line_ratio_prior



"""
This script shows examples of more advanced usage of NebulaBayes.

There are examples of how to:
 - Filter a model grid table to consider a smaller parameter space than the
   full grid
 - Include an upper bound
 - Include a custom prior
 - Use a custom plotting callback to add annotations to plots
 - Use the built-in dereddening capability
 - Retrieve information on the parameter estimates and comparison of the
   "best model" from output tables using the NebulaBayes API.

This script may be run unchanged if it is still in the NebulaBayes/docs
directory.  Otherwise (assuming NebulaBayes is installed) remove the three
lines ending with "sys.path.insert..." above, and edit the "OUT_DIR" below.
The outputs are saved in the NebulaBayes/docs directory by default.
"""
OUT_DIR = DOCS_PATH



##############################################################################
print("\nRunning filtered grid, prior callback, plot callback example...")

def filter_HII_grid():
    """
    Filter the built-in HII grid to reduce the covered parameter space.
    """
    # First load the binary grid table and convert to a pandas DataFrame table
    grid_table_file = os.path.join(NB_PARENT_DIR, "grids",
                                    "NB_HII_grid.fits.gz")
    BinTableHDU_0 = fits.getdata(grid_table_file, 0)
    DF_grid = Table(BinTableHDU_0).to_pandas()

    # Fix log P/k to the value log10 P/k = 6.6, reducing the grid from
    # 3 dimensions to 2 (remaining parameters: "log U" and "12 + log O/H")
    DF_grid = DF_grid[DF_grid["log P/k"] == 6.6]

    # Remove the lowest six oxygen abundances
    abunds = np.unique(DF_grid["12 + log O/H"].values)  # A sorted array
    DF_grid = DF_grid[DF_grid["12 + log O/H"].isin(abunds[6:])]

    return DF_grid



DF_grid = filter_HII_grid()
grid_params = ["12 + log O/H", "log U"]  # log P/k was removed
interp_shape = [160, 80]  # Interpolated points along each dimension
linelist = ["OII3726_29", "Hbeta", "OIII5007", "OI6300", "Halpha", "NII6583",
            "SII6716", "SII6731"]
obs_fluxes = [1.225,  1,    0.4494, 0.02923, 4.251,  1.653,   0.4560,  0.4148]
obs_errs =  [0.003, 0.0017, 0.0012, 0.00052, 0.0027, 0.00173, 0.00102, 0.00099]
# Must supply wavelengths for dereddening; see NebulaBayes/grids/Linelist.csv
wavelengths = [3727, 4861, 5007, 6300, 6562.8, 6583, 6716.4, 6731]  # Angstroem

# Set up interpolated grids.  Arbitrarily we set grid_error=0.5, to have a very
# large systematic fractional grid flux error.
NB_Model_1 = NB_Model(DF_grid, grid_params, linelist,
                      interpd_grid_shape=interp_shape, grid_error=0.5)



def calculate_custom_prior(DF_obs, grids_dict, grid_spec, grid_rel_err):
    """
    Example callback function passed to NebulaBayes to calculate a custom prior.

    Calculate a prior PDF over the n-D interpolated grid, combining a
    contribution from a line-ratio prior with a contribution from information
    on a particular grid parameter (the ionisation parameter).

    This function shows the required inputs and outputs.

    DF_obs:     A pandas DataFrame table holding the observed fluxes/errors.
                There is a row for each emission line.
    grids_dict: Dictionary that maps line names to nD interpolated flux arrays.
                These interpolated model flux arrays haven't been normalised
                to the "norm_line" yet.
    grid_spec:  An NB1_Process_grids.Grid_description instance holding basic
                information for the interpolated grids, for example in the
                "param_names", "param_values_arrs" and "shape" attributes.
    grid_rel_err: The systematic relative error on grid fluxes, as a linear
                  proportion (between 0 and 1).

    Return a numpy array of the value of the prior over the full interpolated
    parameter space.
    """
    # Firstly calculate a contribution to the prior using the NebulaBayes
    # "line-ratio prior" feature, using the metallicity-sensitive N2O2 ratio
    contribution_N2O2 = calculate_line_ratio_prior(DF_obs, grids_dict,
                           grid_rel_err, line_1="NII6583", line_2="OII3726_29")
    # "contribution_N2O2" is an array over the full n-D interpolated grid.
    
    # Next calculate a contribution to the prior that varies only with the
    # ionisation parameter, log U.
    # Find the index of the "log U" grid parameter (each parameter corresponds
    # to a grid dimension; this is the index of that grid dimension):
    param_ind = grid_spec.param_names.index("log U")
    # The list of interpolated parameter values for the "log U" grid parameter:
    all_U_values = grid_spec.param_values_arrs[param_ind]  # Sorted 1D array

    # Construct a prior on log U which is uniform in log space above
    # log U = -3.0, and rapidly decreasing (in a half-Gaussian) below this cut
    # (this is just an example and isn't very physically motivated)
    cut = -3.0  # (log space)
    log_U_1D_prior = np.exp( -((all_U_values - cut) / 0.2)**2 / 2.)
    log_U_1D_prior[all_U_values >= cut] = 1.0
    # This array has only one dimension.  Construct a slice to use numpy
    # "broadcasting" to apply the 1D prior over the whole 2D grid:
    # (The following method works in nD, although here it's a 2D grid)
    slice_NLR = [np.newaxis for _ in grid_spec.shape]
    slice_NLR[param_ind] = slice(None)  # "[slice(None)]" means "[:]"
    slice_NLR = tuple(slice_NLR)
    contribution_U = np.ones(grid_spec.shape) * log_U_1D_prior[slice_NLR]

    # Combine the two contributions to the prior
    prior = contribution_U * contribution_N2O2

    # The prior PDF will be properly normalised by NebulaBayes later
    return prior



def plotting_callback(out_filename, fig, axes, Plotter, config_dict):
    """
    A function passed to NebulaBayes to modify the posterior plot before it
    is saved.

    NebulaBayes calls this callback function after all the plotting actions
    have been performed, instead of saving the figure to out_filename.  Hence
    this function needs to do the saving.

    out_filename: Filename of output file.  A plotting callback is only used
            if the relevant output plot(s) are being produced, which means the
            output file/directory must have been specified by the user.
    fig: The matplotlib.figure.Figure object
    axes: The 2D array of matplotlib axes
    Plotter: The NebulaBayes NB4_Plotting.ND_PDF_Plotter object which is
            currently doing the plotting
    config_dict: The relevant dict from the list "plot_configs" argument
            supplied when running NebulaBayes (or the default config dict).

    No return value is required.
    """
    # Add an annotation to an axes - a label and an arrow.
    # Due to the mysteries of matplotlib, only "figure fraction" coordinates
    # seem to work.
    axes[0,0].annotate("Highest probability density", xy=(0.42, 0.22),
                       xycoords="figure fraction", xytext=(0.18, 0.31),
                       textcoords="figure fraction", color="yellow", size=8,
                       arrowprops=dict(arrowstyle="-", linewidth=1.5,
                       fc="yellow", color="yellow"))

    # Add a title to the figure
    axes[0,0].annotate("Posterior", xy=(0.69, 0.95),
                       xycoords="figure fraction", size=14)

    # Add another axes to the figure, just for fun
    ax_1 = plt.axes( [0.73, 0.73, 0.15, 0.15] )
                   # [left, bottom, width, height] in figure fraction coords
    ax_1.plot([1,2], [3,4])
    ax_1.set_xticks([])

    # Save out the modified plot
    fig.savefig(out_filename)



# Configure options for a NB run:
kwargs = {"posterior_plot": os.path.join(OUT_DIR,"2_posterior_plot.pdf"),
          "prior_plot": os.path.join(OUT_DIR, "2_prior_plot.pdf"),
          "estimate_table": os.path.join(OUT_DIR, "2_param_estimates.csv"),
          "prior": calculate_custom_prior,  # The callback function
          "deredden": True,  # Match model Balmer decrement everywhere in grid
          "obs_wavelengths": wavelengths,  # Needed for dereddening
          "norm_line": "Hbeta",  # Obs and model fluxes normalised to Hbeta
          "param_display_names": {"log U": r"$\log \; U$",
                                  "12 + log O/H": r"$12 + \log \; $O/H"},
          "plot_configs": [{}, {}, {}, {}], # Prior, like, posterior, per-line
          }
# Add "Best model" table to prior plot:
kwargs["plot_configs"][0]["table_on_plot"] = True
# Use our custom callback function when making the posterior plot:
kwargs["plot_configs"][2]["callback"] = plotting_callback

# Do parameter estimation once:
Result = NB_Model_1(obs_fluxes, obs_errs, linelist, **kwargs)


# In the results, looking at the prior plot we can see that the N2O2 and log U
# priors worked almost independently.  Looking at the posterior, the posterior
# is well constrained, despite the massive relative grid error.

# The NB_Model instance can be called repeatedly to do Bayesian parameter
# estimation on different sets of observed fluxes with the same grid (which
# only needs to be interpolated once).
# You can access all the NebulaBayes internal data, PDFs, results, etc. on the
# Result object.



# Now we extract some data:




##############################################################################
print("\nRunning upper bound example...")

# Example of use of upper bound
linelist1 = ["OII3726_29", "Hbeta", "OIII5007", "OI6300", "Halpha", "NII6583",
            "SII6716", "SII6731"]
# Include an upper bound on OI6300, signaled by the -inf placeholder:
obs_fluxes1 = [1.225,  1,    0.4494, -np.inf, 4.251,  1.653,   0.4560,  0.4148]
obs_errs1 =  [0.003, 0.0017, 0.0012, 0.04,   0.0027, 0.00173, 0.00102, 0.00099]
kwargs1 = {}  # Use defaults - don't write outputs or deredden
Result1 = NB_Model_1(obs_fluxes1, obs_errs1, linelist1, **kwargs1)



print("Advanced example script complete.")
