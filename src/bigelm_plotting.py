from __future__ import print_function, division
import numpy as np  # Core numerical library
import matplotlib.pyplot as plt  # Plotting
# For generating a custom colourmap:
# from matplotlib.colors import LinearSegmentedColormap
# For finding Cartesian product and combinatorial combinations:
# import itertools



"""
Adam D. Thomas 2015 - 2016

"""


# #============================================================================
# def plot_marginalised_posterior(image_out, Params, Raw_grids, Interpd_grids, 
#                                 marginalised_posteriors_1D, marginalised_posteriors_2D):
#     """
#         Generate a plot of all the 2D and 1D marginalised posterior pdfs.
#         Plot the results as a "corner plot", a triangular grid of
#         density images for each 2D marginalised posterior, with
#         appropriate 1D plots of 1D marginalised posteriors included
#         at the end of each row/column.

#         This function will only be called if the user provided a
#         filename for the output image.
#     """

#     print("Plotting marginalised posteriors...")
#     # Create a figure and a 2D-array of axes objects:
#     fig, axes = plt.subplots( Params.n_params, Params.n_params,
#                               figsize=(1.0+Params.n_params*2.0, 1.0+Params.n_params*2.0) ) # figsize in inches
#     # Flip array of axes so the images will fill the lower-left half of the subplot grid:
#     axes = np.flipud(np.fliplr(axes))
#     for ax in axes.ravel():
#         ax.set_visible(False)  # Turn all axes off for now.  Needed axes will be turned on later.
    
#     # Make a custom colourmap - black to white through green, as in Blanc+ 2015
#     # Use a list of RGB tuples (values normalised to [0,1])
#     cmap1 = LinearSegmentedColormap.from_list( "cmap1", # Don't think name is necessary...?
#                                                [(0,0,0),(56./255,132./255,0),(1,1,1)] )
#     plt.register_cmap(cmap=cmap1)  # So we can use the custom colourmap name


#     # Iterate over the 2D marginalised posteriors:
#     for double_name, param_inds_double in zip(Params.double_names,
#                                               Params.double_indices):
#         # We will plot an image for each marginalised posterior
#         ind_y, ind_x = param_inds_double
#         # The first parameter is on the y-axis; the second is on the x-axis.
#         # ind_y (ind_x) is the index of the y-axis (x_axis) parameter in
#         # the list of parameters Params.names
#         # Note that here ind_y ranges from 0 to Params.n_params-2, and
#         # ind_x ranges from 1 to Params.n_params-1.
#         ax_i = axes[ ind_y, ind_x ]
#         ax_i.set_visible(True)  # Turn this axis back on

#         # Calculate the image extent [min_x, max_x, min_y, max_y]:
#         image_extent = [ np.min( Interpd_grids.val_arrs[ind_x] ),
#                          np.max( Interpd_grids.val_arrs[ind_x] ),
#                          np.min( Interpd_grids.val_arrs[ind_y] ),
#                          np.max( Interpd_grids.val_arrs[ind_y] )  ]
#         image_extent[0] -= Interpd_grids.spacing[ind_x]/2.
#         image_extent[1] += Interpd_grids.spacing[ind_x]/2.
#         image_extent[2] -= Interpd_grids.spacing[ind_y]/2.
#         image_extent[3] += Interpd_grids.spacing[ind_y]/2.
#         # Note that the image extent specifies the locations of the outer edges
#         # of the outer pixels.

#         # We use a custom image aspect to force a square subplot:
#         image_aspect = 1.0 / ( (image_extent[3] - image_extent[2]) /
#                                (image_extent[1] - image_extent[0])   )

#         # Actually generate the image of the 2D marginalised posterior:
#         ax_i.imshow( marginalised_posteriors_2D[double_name], origin="lower", 
#                      extent=image_extent, vmin=0, interpolation="spline16", 
#                      cmap=cmap1, aspect=image_aspect )
#         # Data point [0,0] is in the bottom-left of the image; the next point above the lower-left corner
#         # is [1,0], and the next point to the right of the lower-left corner is [0,1];
#         # i.e. array indexing is along the lines of marginalised_posteriors_2D[double_name][y_i, x_i]

#         # Plot dots to show the location of gridpoints from the raw model grid:
#         raw_gridpoints_iter = itertools.product(Raw_grids.val_arrs[ind_x],
#                                                 Raw_grids.val_arrs[ind_y])
#         ax_i.scatter(*zip(*raw_gridpoints_iter), marker='o', s=0.3, color='0.4')
        
#         # Format the current axes:
#         ax_i.set_xlim( image_extent[:2] )
#         ax_i.set_ylim( image_extent[2:] )
#         ax_i.tick_params( direction='inout', length=6 ) # Make ticks longer, and extend in and out of axes
#         for tick in ax_i.get_xticklabels():  # Rotate x tick labels
#                 tick.set_rotation(90)
#         if ind_y == 0: # If we're in the first row of plots
#             # Generate y-axis label.  The following is a workaround because a regression in matplotlib
#             # means that the xycoords="axes fraction" option doesn't work outside the axes extent.
#             label_x = (image_extent[1] + image_extent[0])/2.0   # Average
#             label_y = image_extent[2] - (image_extent[3] - image_extent[2])*0.6
#             ax_i.annotate(Params.display_names[Params.names[ind_x]], (label_x, label_y),
#                           xycoords="data", annotation_clip=False, 
#                           horizontalalignment="center", verticalalignment="center")
#         if ind_x == Params.n_params-1: # If we're in the first column of plots
#             # Generate y-axis label.  The following is a workaround because a regression in matplotlib
#             # means that the xycoords="axes fraction" option doesn't work outside the axes extent.
#             label_x = image_extent[0] - (image_extent[1] - image_extent[0])*0.6
#             label_y = (image_extent[3] + image_extent[2])/2.0   # Average
#             ax_i.annotate(Params.display_names[Params.names[ind_y]], (label_x, label_y),
#                           xycoords="data", rotation="vertical", horizontalalignment="center",
#                           verticalalignment="center", annotation_clip=False)


#     # Iterate over the 1D marginalised posteriors:
#     # We plot the 1D pdfs along the diagonal of the grid of plots:
#     for ind, param in enumerate(Params.names):
#         ax_i = axes[ ind, ind ]
#         ax_i.set_visible(True)  # turn this axis back on
#         ax_i.plot(Interpd_grids.val_arrs[ind], marginalised_posteriors_1D[param],
#                   color="black")
#         ax_i.set_yticks([])  # No y-ticks
#         ax_i.set_xlim( np.min( Interpd_grids.val_arrs[ind] ) - Interpd_grids.spacing[ind]/2.,
#                        np.max( Interpd_grids.val_arrs[ind] ) + Interpd_grids.spacing[ind]/2.  )
#         # Make ticks longer, and extend in and out of axes
#         ax_i.tick_params( direction='inout', length=6 )
#         for tick in ax_i.get_xticklabels():
#                 tick.set_rotation(90)  # Rotate x tick labels

#     # Adjust spacing between and around subplots (spacing in inches):
#     fig.subplots_adjust(left=0.2, bottom=0.2, wspace=0.6, hspace=0.6)

#     print("Saving figure...")
#     fig.savefig( image_out )
#     plt.close(fig)



#============================================================================
def plot_MCMC_chain(sampler_chain,param_display_names, show=False, outfile=None):
    """
    Plot the "chain" of sampled parameter values in a MC Markov Chain
    sampler_chain: array of shape (nwalkers,nsteps,ndim)
    param_display_names: List of names to be displayed for params, in order of
                         the dimensions (i.e. in order of sampler_chain[i,j,:])
    show: Boolean.  Show interactive plot?
    outfile: Filename; default None.
    """
    # Inspired by plot at http://dan.iel.fm/emcee/current/user/line/
    
    nwalkers, nsteps, ndim = sampler_chain.shape

    # Initialise plot and axes:
    fig, ax_arr = plt.subplots(nrows=ndim, ncols=1, sharex=True, sharey=False,
                  squeeze=True, figsize=(4,1.5+ndim*1.5))# Width and height in inches

    x_vec = np.arange(nsteps)
    # Iterate over axes plotting the chains for each parameter
    for i, ax in enumerate(ax_arr):
        # This axes is for the ith parameter (dimension)
        for walker in range(nwalkers):
            ax.plot(x_vec, sampler_chain[walker,:,i], c="0.6", lw=0.5)
        ax.set_ylabel(param_display_names[i])

    ax_arr[-1].set_xlabel("Step number")
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, hspace=0, top=0.95)
    
    if outfile: # Save out image if requested
        plt.savefig(outfile)

    if show: # Show interactive plot if requested
        was_interactive = plt.isinteractive()
        plt.ioff() # Make not interactive - block program from continuing
        print("Displaying figure; close to continue...")
        plt.show()
        plt.interactive(was_interactive) # Return to previous interactive setting



