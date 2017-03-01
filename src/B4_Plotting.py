from __future__ import print_function, division
# from collections import OrderedDict as OD
import numpy as np  # Core numerical library
import matplotlib.pyplot as plt  # Plotting
# For generating a custom colourmap:
from matplotlib.colors import LinearSegmentedColormap
# For finding Cartesian product and combinatorial combinations:
import itertools



"""
Adam D. Thomas 2015 - 2017

"""



def plot_marginalised_ndpdf(out_filename, Bigelm_nd_pdf, Raw_grids, plot_anno=None):
    """
    Generate a corner plot of all the 2D and 1D marginalised pdfs for an
    n-dimensional pdf.  This function may be used for the prior, lkelihood or
    posterior.
    resulting "corner plot" is a triangular grid of 2-D images for each 2D
    marginalised pdf, with appropriate 1D plots of 1D marginalised
    pdfs included along the diagonal.  This function is designed to
    produce attractive plots independent of the dimensionality (axes grid size).
    out_filename: The filename for the output corner plot image file.
    Bigelm_nd_pdf: An object which contains the 1D and 1D marginalised pdfs and
                   interpolated grid information
    Raw_grids: Object holding information about the raw grids, used in plotting
               the original (non-interpolated) grid points.
    plot_anno: Text to annotate the output image in the empty upper-triangle of
               the grid of plots. 
    """
    # Some configuration
    label_fontsize = 7
    tick_fontsize = 7
    tick_size = 3
    label_kwargs = {"annotation_clip":False, "horizontalalignment":"center",
                    "verticalalignment":"center", "fontsize":label_fontsize}
    # Make a custom colourmap for the 2D marginalised pdfs - black to
    # white through green, as in Blanc+ 2015
    # Use a list of RGB tuples (values normalised to [0,1])
    im_cmap = LinearSegmentedColormap.from_list( "cmap1", # Name unnecessary?
                                        [(0,0,0),(56./255,132./255,0),(1,1,1)] )

    # Create a figure and a 2D-array of axes objects:
    # We keep the figure size and bounds of the axes grid the same, and change
    # only n_rows(==n_cols) for different grid dimensions.
    n = Bigelm_nd_pdf.Grid_spec.ndim
    fig_width_ht = 6, 6 # Figure width and height in inches
    grid_bounds = {"left":0.13, "bottom":0.13, "right":0.95, "top":0.95}
    axes_width = (grid_bounds["right"] - grid_bounds["left"]) / n # Figure frac
    axes_height = (grid_bounds["top"] - grid_bounds["bottom"]) / n # Figure frac
    fig, axes = plt.subplots(n, n, figsize=fig_width_ht, gridspec_kw=grid_bounds) 
    # Flip axes array so images fill the lower-left half of the subplot grid:
    axes = np.flipud(np.fliplr(axes))
    # Now axes[0, 0] is the axes in the lower-right.
    for ax in axes.ravel():    # Turn all axes off for now.
        ax.set_visible(False)  # Needed axes will be turned on later.

    # Some quantities for working with the parameters:
    G = Bigelm_nd_pdf.Grid_spec # Interpolated grid description
    par_arr_map = G.paramName2paramValueArr
    interp_spacing = {p : (arr[1] - arr[0]) for p,arr in par_arr_map.items()}
    p_estimates = Bigelm_nd_pdf.DF_estimates["Estimate"] # pandas Series; index is param name

    # Iterate over the 2D marginalised pdfs:
    for double_name, param_inds_double in zip(G.double_names, G.double_indices):
        # We will plot an image for each 2D marginalised pdf
        ind_y, ind_x = param_inds_double
        name_y, name_x = double_name
        # The first parameter is on the y-axis; the second is on the x-axis.
        # ind_y (ind_x) is the index of the y-axis (x_axis) parameter in
        # the list of parameters Params.names
        # Note that here ind_y ranges from 0 to n - 2, and
        # ind_x ranges from 1 to n - 1.
        ax_i = axes[ ind_y, ind_x ]
        ax_i.set_visible(True)  # Turn this axis back on

        # Calculate the image extent:
        x_arr, y_arr = par_arr_map[name_x], par_arr_map[name_y]
        extent = { "xmin" : x_arr.min(), "xmax" : x_arr.max(),
                   "ymin" : y_arr.min(), "ymax" : y_arr.max() }
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

        # Actually generate the image of the 2D marginalised pdf:
        pdf_2D = Bigelm_nd_pdf.marginalised_2D[double_name]
        ax_i.imshow( pdf_2D, vmin=0,
                     origin="lower", extent=extent_list, cmap=im_cmap,
                     interpolation="spline16", aspect=image_aspect )
        # Data point [0,0] is in the bottom-left of the image; the next point
        # above the lower-left corner is [1,0], and the next point to the right
        # of the lower-left corner is [0,1]; i.e. the 2D pdf array indexing is
        # along the lines of marginalised_2D[double_name][y_i, x_i]
        # Uncomment the following to show the x- and y- indices of the axes:
        # ax_i.annotate( "ind_y, ind_x = ({0},{1})".format(ind_y, ind_x),
        #    (0.1,0.5), color="white", xycoords="axes fraction", fontsize=4)

        # Plot dots to show the location of gridpoints from the raw model grid:
        raw_gridpoints_iter = itertools.product(Raw_grids.param_values_arrs[ind_x],
                                                Raw_grids.param_values_arrs[ind_y] )
        ax_i.scatter(*zip(*raw_gridpoints_iter), marker='o', s=0.3, color='0.4')
        
        # Show best estimates (coordinates from peaks of 1D pdf):
        x_best_1d, y_best_1d = p_estimates[name_x], p_estimates[name_y]
        ax_i.scatter(x_best_1d, y_best_1d, marker="o", s=12, facecolor="maroon", linewidth=0)
        # Show peak of 2D pdf:
        max_inds_2d = np.unravel_index(np.argmax(pdf_2D), pdf_2D.shape)
        x_best_2d = x_arr[max_inds_2d[1]]
        y_best_2d = y_arr[max_inds_2d[0]]
        ax_i.scatter(x_best_2d, y_best_2d, marker="v", s=13, facecolor="none", linewidth=0.5, edgecolor="blue")
        # Show projection of peak of full nD pdf:
        max_inds_nd = np.unravel_index(np.argmax(Bigelm_nd_pdf.nd_pdf), Bigelm_nd_pdf.nd_pdf.shape)
        x_best_nd = x_arr[max_inds_nd[ind_x]]
        y_best_nd = y_arr[max_inds_nd[ind_y]]
        ax_i.scatter(x_best_nd, y_best_nd, marker="s", s=21, facecolor="none", linewidth=0.5, edgecolor="orange")


        # Format the current axes:
        ax_i.set_xlim( extent["xmin"], extent["xmax"] )
        ax_i.set_ylim( extent["ymin"], extent["ymax"] )
        ax_i.tick_params( direction='out', length=tick_size )
        if ind_y == 0: # If we're in the first row of plots
            # Generate x-axis label
            label_x = grid_bounds["right"] - axes_width * (ind_x + 0.5)
            label_y = grid_bounds["bottom"] * 0.25
            ax_i.annotate(G.param_display_names[ind_x], (label_x, label_y),
                          xycoords="figure fraction", **label_kwargs)
            for tick in ax_i.get_xticklabels():  # Rotate x tick labels
                    tick.set_rotation(90)
                    tick.set_fontsize(tick_fontsize)
        else: # Not first row
            ax_i.set_xticklabels([]) # No x_labels
        if ind_x == n - 1: # If we're in the first column of plots
            # Generate y-axis label
            label_x = grid_bounds["left"] * 0.25
            label_y = grid_bounds["bottom"] + axes_height * (ind_y + 0.5)
            ax_i.annotate(G.param_display_names[ind_y], (label_x, label_y),
                xycoords="figure fraction", rotation="vertical", **label_kwargs)
            for tick in ax_i.get_yticklabels():
                    tick.set_fontsize(tick_fontsize)
        else: # Not first column
            ax_i.set_yticklabels([]) # No y_labels


    # Iterate over the 1D marginalised pdfs:
    # We plot the 1D pdfs along the diagonal of the grid of plots:
    for ind, param in enumerate(G.param_names):
        ax_i = axes[ ind, ind ]
        ax_i.set_visible(True)  # turn this axis back on
        pdf_1D =  Bigelm_nd_pdf.marginalised_1D[param]
        ax_i.plot(par_arr_map[param], pdf_1D, color="black", zorder=6)
        # Plot a vertical line to show the parameter estimate (peak of 1D pdf)
        y_lim = ax_i.get_ylim()
        ax_i.vlines(p_estimates[G.param_names[ind]], y_lim[0], y_lim[1],
                    linestyles="dashed", color="maroon", lw=0.6, zorder=5)
        ax_i.set_yticks([])  # No y-ticks
        ax_i.set_xlim( np.min( par_arr_map[param] ) - interp_spacing[param]/2.,
                       np.max( par_arr_map[param] ) + interp_spacing[param]/2. )
        ax_i.set_ylim(y_lim)
        if ind == 0: # Last column
            label_x = grid_bounds["right"] - 0.5 * axes_width
            label_y = grid_bounds["bottom"] * 0.25
            ax_i.annotate(G.param_display_names[ind], (label_x, label_y),
                                     xycoords="figure fraction", **label_kwargs)
            for tick in ax_i.get_xticklabels():
                    tick.set_rotation(90)  # Rotate x tick labels
                    tick.set_fontsize(tick_fontsize)
        else: # Not last column
            ax_i.set_xticklabels([]) # No x_labels
        ax_i.tick_params(direction='out', length=tick_size)
    # raise Exception


    # Adjust spacing between and around subplots (spacing in inches):
    fig.subplots_adjust(left=grid_bounds["left"], bottom=grid_bounds["bottom"],
                                                      wspace=0.04, hspace=0.04)

    if plot_anno is not None: # Add text including chisquared and fluxes table
        plt.annotate(plot_anno, (grid_bounds["left"]+n//2*axes_width+0.03, 0.94),
                    xycoords="figure fraction", annotation_clip=False, 
                    horizontalalignment="left", verticalalignment="top", 
                    family="monospace", fontsize=5)

    print("Saving figure...")
    fig.savefig(out_filename)
    plt.close(fig)


