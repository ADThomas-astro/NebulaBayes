from __future__ import print_function, division
import itertools  # For Cartesian product
import sys  # For python version

from matplotlib import __version__ as __mpl_version__
import matplotlib.pyplot as plt  # Plotting
from matplotlib.backends.backend_pdf import PdfPages  # For pdf metadata
# For generating a custom colourmap:
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker  # Customise tick placement
import numpy as np  # Core numerical library
import pandas as pd # For tables ("DataFrame"s)
from ._version import __version__ as __NB_version__
from ._compat import _str_type  # Compatibility


"""
Code for plotting "corner plots" showing marginalised n-dimensional PDFs.

Adam D. Thomas 2015 - 2018
"""



# These are the only allowed "plot_types", which an ND_PDF_Plotter (defined
# below) will be used for. 
plot_types = ["Prior", "Likelihood", "Posterior", "Individual_line"]



class Plot_Config(object):
    """
    Helper class to store the user's plot configuration for a NebulaBayes run.
    """
    # A custom colourmap for the 2D marginalised pdfs - black to white through
    # green, as in Blanc+ 2015
    # Use a list of RGB tuples (values normalised to [0,1])
    default_cmap = LinearSegmentedColormap.from_list( "NB_default",
                                      [(0,0,0), (56./255,132./255,0), (1,1,1)] )
    default_config = { # Include a text "best model" flux comparison table on
                       # the 'corner' plot?
                       "table_on_plot": False,
                       "show_legend": True,  # Show the legend?
                       "legend_fontsize": 4.5,  # Fontsize of labels in legend
                       # The colormap for the images of 2D marginalised PDFs:
                       "cmap": default_cmap,
                       "callback": None,  # Callback to modify plot
                      }
    option_keys = list(default_config.keys())  # List of allowed config keys


    def __init__(self, input_configs):
        """
        Initialise this Plot_Config instance by overriding the defaults with
        the user's inputs.  We also validate the inputs.
        """
        if not isinstance(input_configs, list) or len(input_configs) != 4:
            raise Exception("plot_configs must be a list of length 4")
        # We store a config dict for each of the four plot types
        self.configs = {t:self.default_config.copy() for t in plot_types}
        for plot_type, input_dict in zip(plot_types, input_configs):
            if not isinstance(input_dict, dict):
                raise TypeError("plot_configs list must contain 4 dicts")
            for key, val in input_dict.items(): # input_dict may be empty
                if key not in self.option_keys:
                    raise ValueError("Unknown plot config key " + str(key))
                if key == "callback" and not callable(val):
                    raise TypeError("callback must be a callable")
                self.configs[plot_type][key] = val


    def __getitem__(self, x):
        """ To ease access to the configs """
        return self.configs[x]



Plot_Config_Default = Plot_Config([{}]*4)



def _make_plot_annotation(Plot_Config_1, NB_nd_pdf):
    """
    Make the "best model table" text annotation to include on plots, and store
    it as the "table_for_plot" attribute on Plot_Config_1.  This attribute is
    set to None if the "table_on_plot" option is False (the default).
    """
    pdf_name = NB_nd_pdf.name  # One of the "plot_types" above
    make_anno = ( (Plot_Config_1[pdf_name]["table_on_plot"] is True)
                  and hasattr(NB_nd_pdf, "best_model") )
    if not make_anno:
        Plot_Config_1.table_for_plot = None  # Convenient storage spot
        return
    best_dict = NB_nd_pdf.best_model
    plot_anno = ("Observed fluxes vs. model fluxes at the gridpoint\n"
                 "defined by peaks of the 1D marginalised {0} PDFs\n".format(
                                                             pdf_name.lower()))
    plot_anno += str(best_dict["table"]) + "\n\n"
    plot_anno += r"$\chi^2_r = ${0:.1f}".format(best_dict["chi2"])
    if not isinstance(best_dict["extinction_Av_mag"], _str_type):
        # extinction_Av_mag only calculated when deredden is True,
        # otherwise it's set to the string "NA (deredden is False)"
        plot_anno += "\n" + r"$A_v = ${0:.1f} mag".format(
                                                best_dict["extinction_Av_mag"])
    Plot_Config_1.table_for_plot = plot_anno  # Convenient storage spot



class ND_PDF_Plotter(object):
    """
    Helper class for plotting "corner plots" showing a grid of all possible 2D
    and 1D marginalised PDFs derived from an ND PDF.
    Each instance stores the locations of the raw (un-interpolated) gridpoints,
    so we can overplot the locations of the raw gridpoints without passing in
    raw grid information whenever a plot is produced.

    The plotting method includes many specific constraints (such as how to
    place ticks) that are designed to ensure that NebulaBayes plots look
    consistently good over different versions of matplotlib.
    """
    def __init__(self, raw_gridpts=None):
        """
        Initialise the plotter class, saving an instance attribute to store the
        raw grid parameter values.  We also store some formatting parameters.
        """
        self.raw_gridpts = raw_gridpts # Map of parameter names to lists of raw
                                       # grid parameter values (optional)

        # Some hard-coded plotting configuration
        self.dpi = 200  # Dots per inch image resolution
        self.fs1 = 4.5  # Fontsize of annotation table (if shown)
        self.label_fontsize = 8
        self.tick_fontsize = 7
        self.tick_size = 2
        self.label_kwargs = {"annotation_clip":False,
                  "horizontalalignment":"center", "verticalalignment":"center",
                  "fontsize":self.label_fontsize,
        }
        self.axes_spine_width = 0.8



    def _prepare_fig_and_axes(self, n):
        """
        Create the figure and axes the first time this ND_PDF_Plotter instance
        is called.  On subsequent calls we simply clear the axes to reuse the
        same figure and axes.
        n: The number of dimensions of the grid
        """
        if hasattr(self, "_fig") and hasattr(self, "_axes"):
            # Resuse the saved figure and axes objects
            # This provides a significant speedup compared to making new ones.
            for ax in self._axes.ravel():
                if ax.get_visible():
                    ax.clear()  # Clear images, lines, annotations, and legend
            return

        # Create a new figure and 2D-array of axes objects
        fig_width_ht = (6.0,) * 2  # Figure width and height in inches (equal)
        # We keep the figure size and bounds of the axes grid the same, and
        # change only n_rows(==n_cols) for different grid dimensions.
        gridspec = {"left": 0.13, "bottom": 0.13, "right": 0.98, "top": 0.98,
                    "hspace": 0.02, "wspace": 0.02}
        fig, axes = plt.subplots(n, n, figsize=fig_width_ht,
                                                          gridspec_kw=gridspec)
        
        # Add some more information to the gridspec dict
        gridspec["axes_width"] = (gridspec["right"] - gridspec["left"] -
                           (n - 1) * gridspec["wspace"]) / n  # Figure fraction
        gridspec["axes_height"] = (gridspec["top"] - gridspec["bottom"] -
                           (n - 1) * gridspec["hspace"]) / n  # Figure fraction
        gridspec["n"] = n
        
        # Set up the axes grid
        axes = np.atleast_2d(axes)  # For the n == 1 case
        # Flip axes array so images fill lower-left half of subplot grid:
        axes = np.flipud(np.fliplr(axes))
        # Now axes[0, 0] is the axes in the lower-right.
        for ax in axes.ravel():    # Turn all axes off for now.
            ax.set_visible(False)  # Needed axes will be turned on later.
        
        # Save the figure, axes grid and gridspec dictionary as attributes
        self._axes = axes
        self._fig = fig
        self._gridspec = gridspec



    def _format_2D_PDF_axes(self, ax, extent):
        """
        Set the axis properties and tick properties for an axes displaying a 2D
        marginalised PDF.
        """
        # Set width of axis spine lines
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(self.axes_spine_width)
        # Set tick locations
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=4))
            axis.set_minor_locator(ticker.AutoMinorLocator(n=2))
        # Set which spines ticks are on
        ax.xaxis.set_ticks_position("both")  # On top and bottom spines
        ax.yaxis.set_ticks_position("left")  # On left but not right spine
        # Set the axis limits
        ax.set_xlim(extent["xmin"], extent["xmax"])
        ax.set_ylim(extent["ymin"], extent["ymax"])
        # Manually prune ticks, becuase the "prune" option doesn't work
        # in the MaxNLocator
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        x_cut = xmin + 0.95 * (xmax - xmin)  # Remove ticks above x_cut
        y_cut = ymin + 0.95 * (ymax - ymin)
        xticks, yticks = ax.get_xticks(), ax.get_yticks()
        if len(xticks) > 0 and (xticks[-1] > x_cut):
            # Workaround: use tick values, instead of labels
            xlabels = [round(t, 8) if t < x_cut else "" for t in xticks]
            ax.set_xticklabels(xlabels)
        if len(yticks) > 0 and (yticks[-1] > y_cut):
            # Workaround: use tick values, instead of labels
            ylabels = [round(t, 8) if t < y_cut else "" for t in yticks]
            ax.set_yticklabels(ylabels)
        # Point ticks out of the axes and set their length
        ax.tick_params(direction="out", length=self.tick_size, which="major")
        ax.tick_params(direction="out", length=(0.7 * self.tick_size),
                       which="minor")



    def _add_legend(self, ax_1D, ax_2D, gridspec, fontsize):
        """
        Add a legend to the figure.  The inputs ax_1D and ax_2D are axes which
        have 1- and 2-dimensional PDFs plotted on them, respectively.
        """
        n = gridspec["n"]
        legend_anchor = (  # (x,y) in figure fraction coords
            (gridspec["left"] +
                ((n+1)//2) * (gridspec["axes_width"] + gridspec["wspace"])
                                                                      + 0.005),
            (gridspec["bottom"] +
                (n//2) * (gridspec["axes_height"] + gridspec["hspace"])
                                                                      + 0.00)
                        )
        lh1, ll1 = ax_1D.get_legend_handles_labels()
        lh2, ll2 = ax_2D.get_legend_handles_labels()
        lgd = ax_1D.legend(lh1+lh2, ll1+ll2, loc="lower left", borderpad=1,
                          scatterpoints=1, bbox_to_anchor=legend_anchor, 
                          bbox_transform=self._fig.transFigure,  # Needed
                                    # because we use figure fraction coords
                          fontsize=fontsize, fancybox=False)
        leg_frame = lgd.get_frame()
        leg_frame.set_linewidth(0.5)
        leg_frame.set_edgecolor("black")



    def __call__(self, NB_nd_pdf, out_filename, config=Plot_Config_Default):
        """
        Generate a "corner plot" of all the 2D and 1D marginalised pdfs for an
        n-dimensional pdf.  This method may be used for the prior, lkelihood or
        posterior, or for the individual line PDFs contributing to the likelihood.
        The resulting "corner plot" is a triangular grid of 2-D images for each
        2D marginalised pdf, with appropriate 1D plots of 1D marginalised
        pdfs included along the diagonal.  This method is designed to produce
        attractive plots independent of the dimensionality (axes grid size).
        NB_nd_pdf: An object which contains the 1D and 2D marginalised pdfs and
                   interpolated grid information
        out_filename: The filename for the output corner plot image file
        config: An instance of the Plot_Config class defined above
        """
        # Handle inputs
        plot_type = NB_nd_pdf.name
        assert plot_type in plot_types
        config1 = config[plot_type]
        n = NB_nd_pdf.Grid_spec.ndim
        self._prepare_fig_and_axes(n)
        gridspec = self._gridspec

        # Some quantities for working with the parameters:
        G = NB_nd_pdf.Grid_spec # Interpolated grid description
        par_arr_map = G.paramName2paramValueArr
        interp_spacing = {p : (arr[1] - arr[0]) for p,arr in par_arr_map.items()}
        p_estimates = NB_nd_pdf.DF_estimates["Estimate"]
        # p_estimates is a pandas Series; the index is the parameter name

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
            ax_i = self._axes[ ind_y, ind_x ]
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
            # Note that the image extent specifies the locations of the outer
            # edges of the outer pixels.

            # We use a custom image aspect to force a square subplot:
            image_aspect = 1.0 / ( extent["yrange"] / extent["xrange"] )

            # Actually generate the image of the 2D marginalised pdf:
            pdf_2D = NB_nd_pdf.marginalised_2D[double_name]
            if pdf_2D.min() < 0:  # Ensure the PDF is non-negative
                raise ValueError("The 2D PDF {0} has a negative value!".format(double_name))
            ax_i.imshow( pdf_2D, vmin=0,
                         origin="lower", extent=extent_list, cmap=config1["cmap"],
                         interpolation="spline16", aspect=image_aspect )
            # Data point [0,0] is in the bottom-left of the image; the next point
            # above the lower-left corner is [1,0], and the next point to the right
            # of the lower-left corner is [0,1]; i.e. the 2D pdf array indexing
            # is along the lines of marginalised_2D[double_name][y_i, x_i]
            # Uncomment the following to show the x- and y- indices of the axes:
            # ax_i.annotate( "ind_y, ind_x = ({0},{1})".format(ind_y, ind_x),
            #    (0.1,0.5), color="white", xycoords="axes fraction", fontsize=4)

            if self.raw_gridpts is not None:
                # Plot dots to show the location of gridpoints from the raw model grid:
                raw_gridpoints_iter = itertools.product(self.raw_gridpts[name_x],
                                                        self.raw_gridpts[name_y] )
                ax_i.scatter(*zip(*raw_gridpoints_iter), marker='o', s=0.3, color='0.4')

            # Show best estimates (coordinates from peaks of 1D pdf):
            x_best_1d, y_best_1d = p_estimates[name_x], p_estimates[name_y]
            ax_i.scatter(x_best_1d, y_best_1d, marker="o", s=12, linewidth=0,
                 facecolor="maroon", label="Model defined by peaks of 1D PDFs")
            # Show peak of 2D pdf:
            max_inds_2d = np.unravel_index(np.argmax(pdf_2D), pdf_2D.shape)
            x_best_2d = x_arr[max_inds_2d[1]]
            y_best_2d = y_arr[max_inds_2d[0]]
            ax_i.scatter(x_best_2d, y_best_2d, marker="v", s=13, linewidth=0.5,
                         facecolor="none", edgecolor="blue",
                         label="Peak of 2D marginalised PDF")
            # Show projection of peak of full nD pdf:
            max_inds_nd = np.unravel_index(np.argmax(NB_nd_pdf.nd_pdf),
                                           NB_nd_pdf.nd_pdf.shape)
            x_best_nd = x_arr[max_inds_nd[ind_x]]
            y_best_nd = y_arr[max_inds_nd[ind_y]]
            ax_i.scatter(x_best_nd, y_best_nd, marker="s", s=21, linewidth=0.5,
                         facecolor="none", edgecolor="orange",
                         label="Projected peak of full nD PDF")

            self._format_2D_PDF_axes(ax_i, extent)

            if ind_y == 0: # If we're in the first row of plots
                # Generate x-axis label
                label_x = (gridspec["right"] - ind_x * gridspec["wspace"] -
                                        (ind_x + 0.5) * gridspec["axes_width"])
                label_y = gridspec["bottom"] * 0.25
                ax_i.annotate(G.param_display_names[ind_x], (label_x, label_y),
                              xycoords="figure fraction", **self.label_kwargs)
                for tick in ax_i.get_xticklabels():  # Rotate x tick labels
                        tick.set_rotation(90)
                        tick.set_fontsize(self.tick_fontsize)
            else: # Not first row
                ax_i.set_xticklabels([]) # No x_labels
            if ind_x == n - 1: # If we're in the first column of plots
                # Generate y-axis label
                label_x = gridspec["left"] * 0.25
                label_y = (gridspec["bottom"] + ind_y * gridspec["hspace"] +
                                       (ind_y + 0.5) * gridspec["axes_height"])
                ax_i.annotate(G.param_display_names[ind_y], (label_x, label_y),
                              xycoords="figure fraction", rotation="vertical",
                              **self.label_kwargs)
                for tick in ax_i.get_yticklabels():
                        tick.set_fontsize(self.tick_fontsize)
            else: # Not first column
                ax_i.set_yticklabels([]) # No y_labels

        # Iterate over the 1D marginalised pdfs:
        # We plot the 1D pdfs along the diagonal of the grid of plots:
        for ind, param in enumerate(G.param_names):
            ax_k = self._axes[ ind, ind ]
            ax_k.set_visible(True)  # turn this axis back on
            for axis in ["top", "bottom", "left", "right"]:
                ax_k.spines[axis].set_linewidth(self.axes_spine_width)
            for axis in [ax_k.xaxis, ax_k.yaxis]:
                axis.set_major_locator(ticker.MaxNLocator(nbins=8, min_n_ticks=4))
                axis.set_minor_locator(ticker.AutoMinorLocator(n=2))
            pdf_1D =  NB_nd_pdf.marginalised_1D[param]
            if pdf_1D.min() < 0:  # Ensure the PDF is non-negative
                raise ValueError("The 1D PDF {0} has a negative value!".format(param))
            ax_k.plot(par_arr_map[param], pdf_1D, color="black", linewidth=0.9,
                      zorder=6)
            # Plot a vertical line to show the parameter estimate (peak of 1D pdf)
            y_lim = (0, 1.14*pdf_1D.max())
            if y_lim[1] == 0:
                y_lim = (0, 1)  # If pdf_1D is all zeros
            if plot_type == "Posterior":
                label1 = "Parameter estimate: peak of 1D\nmarginalised PDF"
            else:
                label1 = "Peak of 1D marginalised PDF"
            ax_k.plot([p_estimates[G.param_names[ind]]]*2, y_lim, lw=0.6,
                        linestyle='--', dashes=(3, 1.4), color="maroon",
                        zorder=5, label=label1)
            ax_k.set_yticks([])  # No y-ticks
            ax_k.set_xlim(np.min(par_arr_map[param]) - interp_spacing[param]/2.,
                          np.max(par_arr_map[param]) + interp_spacing[param]/2. )
            ax_k.set_ylim(y_lim[0], y_lim[1])
            if ind == 0: # Last column
                label_x = gridspec["right"] - 0.5 * gridspec["axes_width"]
                label_y = gridspec["bottom"] * 0.25
                ax_k.annotate(G.param_display_names[ind], (label_x, label_y),
                                xycoords="figure fraction", **self.label_kwargs)
                for tick in ax_k.get_xticklabels():
                        tick.set_rotation(90)  # Rotate x tick labels
                        tick.set_fontsize(self.tick_fontsize)
            else: # Not last column
                ax_k.set_xticklabels([]) # No x_labels
            ax_k.tick_params(direction="out", length=self.tick_size, which="major")
            ax_k.tick_params(direction="out", length=0.7*self.tick_size, which="minor")
            ax_k.xaxis.tick_bottom()  # Ticks on bottom but not top spine
            ax_k.yaxis.tick_left()    # Ticks on left but not right spine

        if config1["show_legend"] is True and n > 1:
            # Add legend to current axes.  Legend not required for n = 1.
            self._add_legend(ax_k, ax_i, gridspec,
                             fontsize=config1["legend_fontsize"])

        # Add fluxes table and chisquared as text annotation if requested
        if config1["table_on_plot"] is True:
            anno_location = [gridspec["left"] + (n//2 * (gridspec["axes_width"]
                                + gridspec["wspace"]) + 0.01), gridspec["top"]]
            if n == 1:
                anno_location[1] -= 0.02  # Slightly lower
            if plot_type != "Individual_line":  # If table available
                pd.set_option("display.precision", 4)
                ax_k.annotate(config.table_for_plot, anno_location,
                            xycoords="figure fraction", annotation_clip=False, 
                            horizontalalignment="left", verticalalignment="top", 
                            family="monospace", fontsize=self.fs1)

        if out_filename.endswith(".pdf"):  # Add metadata if output is a .pdf
            Pdf_fig_1 = PdfPages(out_filename)  # Pdf_fig_1 will be closed later
            metadata_dict = Pdf_fig_1.infodict()
            metadata_dict["Creator"] = (metadata_dict["Creator"] + " ; " +
                            "NebulaBayes {0} ; matplotlib {1} ; numpy {2} ; "
                            "python {3}".format(__NB_version__, __mpl_version__,
                            np.__version__, sys.version))

        # Call the user's "callback" function, if it was supplied, otherwise
        # we save the figure
        callback = config1["callback"]
        if callback is not None:
            callback(out_filename, self._fig, self._axes, self, config1)
            # To be safe, delete the figure and axes and regenerate next time
            # The callback may have modified them in undesirable ways
            plt.close(self._fig)
            del self._fig
            del self._axes
        else:  # Save the figure
            if out_filename.endswith(".pdf"):
                plt.savefig(Pdf_fig_1, format="pdf", dpi=self.dpi)
                Pdf_fig_1.close()
            else:
                self._fig.savefig(out_filename, dpi=self.dpi)


