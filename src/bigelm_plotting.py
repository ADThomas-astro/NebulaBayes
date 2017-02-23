from __future__ import print_function, division
from collections import OrderedDict as OD
import numpy as np  # Core numerical library
import matplotlib.pyplot as plt  # Plotting
# For generating a custom colourmap:
from matplotlib.colors import LinearSegmentedColormap
# For finding Cartesian product and combinatorial combinations:
import itertools



"""
Adam D. Thomas 2015 - 2016

"""


#============================================================================
def plot_marginalised_posterior(image_out, Raw_grids, Result, plot_anno=None):
    """
    Generate a plot of all the 2D and 1D marginalised posterior pdfs.
    Plot the results as a "corner plot", a triangular grid of
    density images for each 2D marginalised posterior, with
    appropriate 1D plots of 1D marginalised posteriors included
    at the end of each row/column.

    This function will only be called if the user provided a
    filename for the output image.
    """

    print("Plotting marginalised posteriors...")
    debug = False
    n = Result.Params.n_params
    # Create a figure and a 2D-array of axes objects:
    # We keep the figure size and bounds of the axes grid the same, and change
    # only n_rows(==n_cols) for different grid dimensions.
    #fig_width, fig_height = (0.8 + n*1.8, 0.8 + n*1.8)#  figsize in inches
    fig_width, fig_height = 6, 6 # inches
    bounds = {"left":0.13, "bottom":0.13, "right":0.83, "top":0.83}
    fig, axes = plt.subplots( n, n, figsize=(fig_width, fig_height), gridspec_kw=bounds) 
    # Flip array of axes so the images will fill the lower-left half of the subplot grid:
    axes = np.flipud(np.fliplr(axes))
    for ax in axes.ravel():
        ax.set_visible(False)  # Turn all axes off for now.  Needed axes will be turned on later.
    
    # Make a custom colourmap - black to white through green, as in Blanc+ 2015
    # Use a list of RGB tuples (values normalised to [0,1])
    cmap1 = LinearSegmentedColormap.from_list( "cmap1", # Don't think name is necessary...?
                                               [(0,0,0),(56./255,132./255,0),(1,1,1)] )
    plt.register_cmap(cmap=cmap1)  # So we can use the custom colourmap name

    par_arr_map = OD([(p,v) for p,v in zip(Result.Params.names, Result.val_arrs)])

    # Iterate over the 2D marginalised posteriors:
    for double_name, param_inds_double in zip(Result.Params.double_names,
                                              Result.Params.double_indices):
        # We will plot an image for each marginalised posterior
        ind_y, ind_x = param_inds_double
        name_y, name_x = double_name
        # The first parameter is on the y-axis; the second is on the x-axis.
        # ind_y (ind_x) is the index of the y-axis (x_axis) parameter in
        # the list of parameters Params.names
        # Note that here ind_y ranges from 0 to n - 2, and
        # ind_x ranges from 1 to n - 1.
        ax_i = axes[ ind_y, ind_x ]
        ax_i.set_visible(True)  # Turn this axis back on

        interp_spacing = {p:(v[1]-v[0]) for p,v in par_arr_map.items()}
        # Calculate the image extent:
        extent = { "xmin" : np.min( par_arr_map[name_x] ),
                   "xmax" : np.max( par_arr_map[name_x] ),
                   "ymin" : np.min( par_arr_map[name_y] ),
                   "ymax" : np.max( par_arr_map[name_y] )  }
        extent["xmin"] -= interp_spacing[name_x]/2.
        extent["xmax"] += interp_spacing[name_x]/2.
        extent["ymin"] -= interp_spacing[name_y]/2.
        extent["ymax"] += interp_spacing[name_y]/2.
        extent["xrange"] = extent["xmax"] - extent["xmin"]
        extent["yrange"] = extent["ymax"] - extent["ymin"]
        extent_list = [extent[l] for l in ("xmin", "xmax", "ymin", "ymax")]
        # Note that the image extent specifies the locations of the outer edges
        # of the outer pixels.

        # We use a custom image aspect to force a square subplot:
        image_aspect = 1.0 / ( extent["yrange"] / extent["xrange"] )

        # Actually generate the image of the 2D marginalised posterior:
        ax_i.imshow( Result.posteriors_marginalised_2D[double_name], origin="lower", 
                     extent=extent_list, vmin=0, interpolation="spline16", 
                     cmap=cmap1, aspect=image_aspect )
        # Data point [0,0] is in the bottom-left of the image; the next point above the lower-left corner
        # is [1,0], and the next point to the right of the lower-left corner is [0,1];
        # i.e. array indexing is along the lines of posteriors_marginalised_2D[double_name][y_i, x_i]

        # Plot dots to show the location of gridpoints from the raw model grid:
        raw_gridpoints_iter = itertools.product(Raw_grids.val_arrs[ind_x],
                                                Raw_grids.val_arrs[ind_y])
        ax_i.scatter(*zip(*raw_gridpoints_iter), marker='o', s=0.3, color='0.4')
        
        # Format the current axes:
        ax_i.set_xlim( extent["xmin"], extent["xmax"] )
        ax_i.set_ylim( extent["ymin"], extent["ymax"] )
        ax_i.tick_params( direction='out', length=4 ) # Make ticks longer, and extend out of axes
        if ind_y == 0: # If we're in the first row of plots
            # Generate x-axis label.  The following is a workaround because a regression in matplotlib
            # means that the xycoords="axes fraction" option doesn't work outside the axes extent.
            label_x = (extent["xmax"] + extent["xmin"])/2.0   # Average
            label_y = extent["ymin"] - extent["yrange"]*0.3
            ax_i.annotate(Result.Params.display_names[Result.Params.names[ind_x]],
                          (label_x, label_y), fontsize=7,
                          xycoords="data", annotation_clip=False, 
                          horizontalalignment="center", verticalalignment="center")
            for tick in ax_i.get_xticklabels():  # Rotate x tick labels
                    tick.set_rotation(90)
                    tick.set_fontsize(9)
        elif not debug: # Not first row and not debug mode
            ax_i.set_xticklabels([]) # No x_labels
        if ind_x == n - 1: # If we're in the first column of plots
            # Generate y-axis label.  The following is a workaround because a regression in matplotlib
            # means that the xycoords="axes fraction" option doesn't work outside the axes extent.
            label_x = extent["xmin"] - extent["xrange"]*0.3
            label_y = (extent["ymax"] + extent["ymin"])/2.0   # Average
            ax_i.annotate(Result.Params.display_names[Result.Params.names[ind_y]],
                          (label_x, label_y), fontsize=7,
                          xycoords="data", rotation="vertical", horizontalalignment="center",
                          verticalalignment="center", annotation_clip=False)
            for tick in ax_i.get_yticklabels():
                    tick.set_fontsize(9)
        elif not debug: # Not first columns and not debug mode
            ax_i.set_yticklabels([]) # No y_labels


    # Iterate over the 1D marginalised posteriors:
    # We plot the 1D pdfs along the diagonal of the grid of plots:
    for ind, param in enumerate(Result.Params.names):
        ax_i = axes[ ind, ind ]
        ax_i.set_visible(True)  # turn this axis back on
        ax_i.plot(par_arr_map[param], Result.posteriors_marginalised_1D[param],
                  color="black")
        ax_i.set_yticks([])  # No y-ticks
        ax_i.set_xlim( np.min( par_arr_map[param] ) - interp_spacing[param]/2.,
                       np.max( par_arr_map[param] ) + interp_spacing[param]/2.  )
        if ind == 0: # Last column
            label_x = np.mean(ax_i.get_xlim())
            y_min, y_max = ax_i.get_ylim()
            label_y = y_min - (y_max - y_min)*0.6
            ax_i.annotate(Result.Params.display_names[Result.Params.names[ind]],
                          (label_x, label_y),
                          xycoords="data", annotation_clip=False, 
                          horizontalalignment="center", verticalalignment="center")
            for tick in ax_i.get_xticklabels():
                    tick.set_rotation(90)  # Rotate x tick labels
                    tick.set_fontsize(9)
        elif not debug: # Not last column and not debug mode
            ax_i.set_xticklabels([]) # No x_labels

        # Make ticks longer, and extend out of axes
        ax_i.tick_params( direction='out', length=4 )

    # Adjust spacing between and around subplots (spacing in inches):
    if debug:
        fig.subplots_adjust(left=0.2, bottom=0.2, wspace=0.6, hspace=0.6)
    else:
        fig.subplots_adjust(left=0.15, bottom=0.15, wspace=0.1, hspace=0.1)

    if plot_anno is not None:
        plt.annotate(plot_anno, (0.54, 0.94),
                    xycoords="figure fraction", annotation_clip=False, 
                    horizontalalignment="left", verticalalignment="top", 
                    family="monospace", fontsize=6)

    print("Saving figure...")
    fig.savefig( image_out )
    plt.close(fig)



# #============================================================================
# def plot_MCMC_chain(sampler_chain,param_display_names, show=False, outfile=None):
#     """
#     Plot the "chain" of sampled parameter values in a MC Markov Chain
#     sampler_chain: array of shape (nwalkers,nsteps,ndim)
#     param_display_names: List of names to be displayed for params, in order of
#                          the dimensions (i.e. in order of sampler_chain[i,j,:])
#     show: Boolean.  Show interactive plot?
#     outfile: Filename; default None.
#     """
#     # Inspired by plot at http://dan.iel.fm/emcee/current/user/line/
    
#     nwalkers, nsteps, ndim = sampler_chain.shape

#     # Initialise plot and axes:
#     fig, ax_arr = plt.subplots(nrows=ndim, ncols=1, sharex=True, sharey=False,
#                   squeeze=True, figsize=(4,1.5+ndim*1.5))# Width and height in inches

#     x_vec = np.arange(nsteps)
#     # Iterate over axes plotting the chains for each parameter
#     for i, ax in enumerate(ax_arr):
#         # This axes is for the ith parameter (dimension)
#         for walker in range(nwalkers):
#             ax.plot(x_vec, sampler_chain[walker,:,i], c="0.6", lw=0.5)
#         ax.set_ylabel(param_display_names[i])

#     ax_arr[-1].set_xlabel("Step number")
#     plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, hspace=0, top=0.95)
    
#     if outfile: # Save out image if requested
#         plt.savefig(outfile)

#     if show: # Show interactive plot if requested
#         was_interactive = plt.isinteractive()
#         plt.ioff() # Make not interactive - block program from continuing
#         print("Displaying figure; close to continue...")
#         plt.show()
#         plt.interactive(was_interactive) # Return to previous interactive setting



