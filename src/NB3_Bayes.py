from __future__ import print_function, division
from collections import OrderedDict as OD
import itertools    # For combinatorial combinations
import numpy as np  # Core numerical library
import os.path
import pandas as pd # For tables ("DataFrame"s)
from scipy.integrate import cumtrapz, simps
from scipy.signal import argrelextrema
from .dereddening import deredden as do_dereddening
from .NB1_Process_grids import Grid_description
from .NB2_Prior import calculate_prior


"""
Code to calculate the likelihood and posterior over an N-D grid, marginalise
pdfs to 1D and 2D marginalised pdfs, and generally do Bayesian parameter
estimation.
This module defines two custom NebulaBayes classes: NB_nd_pdf and NB_Result.

Adam D. Thomas 2015 - 2017
"""



class NB_nd_pdf(object):
    """
    Class to hold an N-dimensional PDF and its 1D and 2D marginalised forms.
    Methods on this class marginalise the PDF and perform parameter estimation.
    An instance of this class is used for each of the likelihood, prior and
    posterior.
    The actual nd_pdf array (an attribute of instances of this class) represents
    a probability distribution, i.e. dP = PDF(x) * dx, where the nd_pdf is an
    array of samples of the PDF.
    """
    def __init__(self, nd_pdf, NB_Result, Interpd_grids, DF_obs=None):
        """
        Initialise an instance of the NB_nd_pdf class.
        nd_pdf: A numpy ndarray holding the (linear) pdf.
        NB_Result: A NB_Result object (defined in this module)
        Interpd_grids: A NB_Grid object (defined in NB1_Process_grids.py)
                       holding the interpolated model grid fluxes and the
                       description of the model grid.
        DF_obs: A pandas DataFrame table holding the observed emission line
                fluxes and errors.  The "best model table" is only generated if
                this is supplied.
        The nd_pdf will be normalised.
        """
        if np.any(~np.isfinite(nd_pdf)):
            raise ValueError("The nd_pdf is not entirely finite")
        if nd_pdf.min() < 0:
            raise ValueError("The nd_pdf contains a negative value")
        self.nd_pdf = nd_pdf
        self.Grid_spec = NB_Result.Grid_spec
        # Add self.marginalised_2d and self.marginalised_1d attributes and
        # normalise the self.nd_pdf attribute:
        self.marginalise_pdf()
        # Make a parameter estimate table based on this nd_pdf
        self.make_parameter_estimate_table() # add self.DF_estimates attribute
        if DF_obs is not None:
            # Make a table comparing model and observed fluxes for the 'best' model
            self.make_best_model_table(DF_obs, Interpd_grids, NB_Result)
            # We added the self.DF_best attribute
            # Calculate chi2 for the 'best' model (add self.chi2 attribute):
            self.calculate_chi2(NB_Result.deredden)



    def marginalise_pdf(self):
        """
        Calculate normalised 1D and 2D marginalised pdfs for all possible
        combinations of parameters.
        """
        #print("Calculating 2D and 1D marginalised pdfs...")

        # The interpolated grids have uniform spacing:
        spacing = [(v[1] - v[0]) for v in self.Grid_spec.param_values_arrs]
        n = self.Grid_spec.ndim
        #--------------------------------------------------------------------------
        # Calculate the 2D marginalised pdf for every possible combination
        # of 2 parameters
        # List of all possible pairs of two parameter names:
        param_names = self.Grid_spec.param_names
        p2ind = self.Grid_spec.paramName2ind
        double_names = list(itertools.combinations(param_names, 2))
        self.Grid_spec.double_names = double_names
        # Looks something like: [('BBBP','Gamma'), ('BBBP','NT_frac'),
        #                        ('BBBP','UH_at_r_inner'), ('Gamma','NT_frac'),
        #                        ('Gamma','UH_at_r_inner'), ('NT_frac','UH_at_r_inner')]
        # Corresponding list of possible combinations of two parameter indices:
        double_indices = [(p2ind[p1], p2ind[p2]) for p1,p2 in double_names]
        self.Grid_spec.double_indices = double_indices
        # Looks something like: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        # Note that the order of indices in each tuple is from smaller to larger
        
        # Initialise dictionary of all possible 2D marginalised pdf arrays:
        marginalised_2D = {}  # The dict keys will be tuples of 2 parameter names.
        # Iterate over all possible pairs of parameters:
        for double_name, param_inds_double in zip(double_names, double_indices):
            # Generate list of indices/dimensions/parameters to integrate over:
            inds_for_integration = np.arange(n).tolist()  # Initialise
            inds_for_integration.remove( param_inds_double[0] )
            inds_for_integration.remove( param_inds_double[1] )
            inds_for_integration.reverse() # Ensure we integrate over higher dimensions first,
            # so dimension index numbers are still valid after each integration.

            working_arr = self.nd_pdf.copy()  # Working array - we'll integrate
            # out the dimensions of this array one dimension at a time.  Keep
            # integrating until the result only has two dimensions:
            for param_index in inds_for_integration:
                # Integrate over this dimension (parameter), using the trapezoidal rule
                working_arr = np.trapz(working_arr, axis=param_index,
                                                        dx=spacing[param_index])
            marginalised_2D[double_name] = working_arr # Store result

        #--------------------------------------------------------------------------
        # Calculate the 1D marginalised pdf for each individual parameter
        # Initialise dictionary of all 1D marginalised pdf arrays:
        marginalised_1D = {}

        # For the first parameter in param_names, integrate the first 2D
        # marginalised pdf over the other dimension (parameter), using Simpson's
        # rule:
        marginalised_1D[param_names[0]] = simps(marginalised_2D[double_names[0]],
                                                        axis=1, dx=spacing[1])
        #    np.trapz(marginalised_2D[double_names[0]], axis=1, dx=spacing[1])

        # For all parameters after the first in param_names:
        for double_name, param_inds_double in zip(double_names[:n-1], double_indices[:n-1]):
            # For each pair of parameters we take the second parameter, and integrate 
            # over the first parameter of the pair (which by construction is always the
            # first parameter in param_names).
            assert param_inds_double[0] == 0 
            param = param_names[param_inds_double[1]]
            # Integrate over first dimension (parameter) using Simpson's rule:
            marginalised_1D[param] = simps(marginalised_2D[double_name], axis=0,
                                                                  dx=spacing[0])
            # marginalised_1D[param] = np.trapz(marginalised_2D[double_name],
            #                                             axis=0, dx=spacing[0])


        #--------------------------------------------------------------------------
        # Calculate the 0D marginalised pdf (by which I mean find
        # the normalisation constant - the 0D marginalised pdf should be 1!)
        # Then normalise the 1D and 2D marginalised PDFs:

        # Firstly find the integral over all n dimensions by picking
        # any 1D marginalised pdf (we use the first) and integrating over it:
        # integral = np.trapz(marginalised_1D[param_names[0]], dx=spacing[0])
        integral = simps(marginalised_1D[param_names[0]], dx=spacing[0])
        # Now actually normalise each 2D and 1D marginalised pdf:
        for double_name in double_names:
            # Divide arrays in-place in memory:
            marginalised_2D[double_name] /= integral
        for param in param_names:
            # Divide arrays in-place in memory:
            marginalised_1D[param] /= integral
        # Now normalise the full pdf.  In the case of the likelihood and prior,
        # this normalisation is important; for the posterior it's probably not.
        self.nd_pdf /= integral

        self.marginalised_2D = marginalised_2D
        self.marginalised_1D = marginalised_1D



    def make_parameter_estimate_table(self):
        """
        Do parameter estimation for all parameters in the grid.  Where the
        nd_pdf is the posterior, this is the proper Bayesian parameter
        estimation.
        Creates a pandas DataFrame with a row for each parameter, stored as the
        attribute self.DF_estimates
        """
        # DataFrame columns and types, in order:
        columns = [("Parameter", np.str),  ("Estimate", np.float),
                   ("CI68_low", np.float), ("CI68_high", np.float),
                   ("CI95_low", np.float), ("CI95_high", np.float),
                   ("Est_in_CI68?", np.str),  ("Est_in_CI95?", np.str),
                   ("Est_at_lower?", np.str), ("Est_at_upper?", np.str),
                   ("P(lower)", np.float),    ("P(upper)", np.float),
                   ("P(lower)>50%?", np.str), ("P(upper)>50%?", np.str),
                   ("n_local_maxima", np.int)]
        n = self.Grid_spec.ndim
        OD1 = OD([(c, np.zeros(n, dtype=t)) for c,t in columns])
        DF_estimates = pd.DataFrame(OD1) # Initialise DataFrame
        DF_estimates.loc[:,"Parameter"] = self.Grid_spec.param_names
        DF_estimates.sort_values(by="Parameter", inplace=True)
        # Sort DF, so the DF is deterministic (note Upper and lower case param
        # names are sorted into separate sections)
        DF_estimates.set_index("Parameter", inplace=True)
        for col in [col for col,t in columns if t == np.float]:
            DF_estimates.loc[:,col] = np.nan
        DF_estimates.loc[:,"n_local_maxima"] = -1
        
        # Fill in DF_estimates: 
        for p, pdf_1D in self.marginalised_1D.items():
            param_val_arr = self.Grid_spec.paramName2paramValueArr[p]
            p_dict = make_single_parameter_estimate(p, param_val_arr, pdf_1D)
            for field, value in p_dict.items():
                if field != "Parameter":
                    DF_estimates.loc[p,field] = value

        self.DF_estimates = DF_estimates



    def make_best_model_table(self, DF_obs, Interpd_grids, NB_Result):
        """
        Make a pandas dataframe comparing observed emission line fluxes with
        model fluxes for the model corresponding to the parameter estimates (the
        'best' model).  The parameter estimates are derived from the 1D
        mariginalised PDF, so note that this 'best' point in the parameter space
        does not necessarily correspond to the peak of any 2D marginalised pdf
        nor to any projection of the peak of the ND pdf to a lower parameter
        space.
        """
        DF_best = DF_obs.copy() # Index: "Line"; columns: "Flux", "Flux_err"
        # DF_obs may also possibly have a "Wavelength" column
        DF_best.rename(columns={"Flux":"Obs"}, inplace=True)

        inds_max = np.unravel_index(self.nd_pdf.argmax(), self.nd_pdf.shape)
        grid_fluxes_max = [Interpd_grids.grids[l][inds_max] for l in DF_best.index]
        DF_best["Model"] = grid_fluxes_max
        
        if NB_Result.deredden: # If we dereddened the observed fluxes at each gridpoint
            obs_flux_dered = [arr[inds_max] for arr in NB_Result.obs_flux_arrs]
            obs_flux_err_dered = [arr[inds_max] for arr in NB_Result.obs_flux_err_arrs]
            DF_best["Obs_dered"] = obs_flux_dered
            DF_best["Flux_err_dered"] = obs_flux_err_dered
            DF_best["Obs_S/N_dered"] = (DF_best["Obs_dered"].values /
                                        DF_best["Flux_err_dered"].values   )
            DF_best["Delta_(SDs)"] = ((DF_best["Model"].values - DF_best["Obs_dered"].values)
                                        / DF_best["Flux_err_dered"].values )
            # Columns to include in output and their order (index is "Line"):
            cols_to_include = ["Model", "Obs_dered", "Obs_S/N_dered", "Delta_(SDs)"]
        else:
            DF_best["Obs_S/N"] = DF_best["Obs"].values / DF_best["Flux_err"].values
            DF_best["Delta_(SDs)"] = ((DF_best["Model"].values - DF_best["Obs"].values)
                                        / DF_best["Flux_err"].values )
            # Columns to include in output and their order (index is "Line"):
            cols_to_include = ["Model", "Obs", "Obs_S/N", "Delta_(SDs)"]
        
        self.DF_best = DF_best[cols_to_include]



    def calculate_chi2(self, deredden):
        """
        Calculate a chi^2 value which describes how well the model corresponding
        to the parameter best estimates matches the observations.
        deredden: Boolean.  Did we deredden the observed line fluxes to match
                  the Balmer decrement at every interpolated model gridpoint?
        """
        DF = self.DF_best # Table comparing observations with "best model"
        # We'll need to reconstruct the error column
        if deredden: # If we dereddened the observed fluxes at each gridpoint
            flux_err = DF["Obs_dered"].values / DF["Obs_S/N_dered"].values
            chi2_working = ( (DF["Obs_dered"].values - DF["Model"].values)**2
                                         / flux_err**2  )
        else:
            flux_err = DF["Obs"].values / DF["Obs_S/N"].values
            chi2_working = ( (DF["Obs"].values - DF["Model"].values)**2
                                         / flux_err**2  )
        chi2 = np.sum(chi2_working)
        grid_n = self.Grid_spec.ndim
        # Degrees of freedom (the normalisation line doesn't count as a data
        # point, since its flux is normalised to 1)
        dof = (len(DF) - 1) - grid_n
        if deredden:
            dof -= 1 # Halpha doesn't count as a data point; the obs fluxes are
                     # dereddened to match the Balmer decrement at every gridpoint
                     # so Halpha observations always match the prediction
        chi2 /= dof # The "reduced chi-squared"
        self.chi2 = chi2



def make_single_parameter_estimate(param_name, val_arr, pdf_1D):
    """
    Bayesian parameter estimate for a single parameter, including the credible
    intervals.  This function is also used to make "estimates" using the prior
    and likelihood pdfs, which may be of interest but isn't full Bayesian
    parameter estimation.
    param_name: String giving name of the parameter.
    val_arr: 1D numpy array of parameter co-ordinate values associated with
             the values listed in pdf_1D
    pdf_1D: 1D numpy array of the marginalised 1D pdf for the parameter
            param_name; the pdf values correspond to the parameter values listed
            in val_arr.
    Returns a dictionary.  See "make_parameter_estimate_table" for contents.
    """
    if np.any(~np.isfinite(pdf_1D)):
        raise ValueError("The 1D pdf for {0} is not all finite".format(param_name))
    if np.any(pdf_1D < 0):
        raise ValueError("The 1D pdf for {0} has a negative value".format(param_name))
    length = val_arr.size
    assert length == pdf_1D.size # Sanity check
    index_array = np.arange(length)
    bool_map = {True:"Y", False:"N"}
    out_dict = {}  # To hold results for this parameter
    
    out_dict["Parameter"] = param_name
    # Calculate estimate of parameter value (location of max in 1D pdf)
    # (This is the Bayesian parameter estimate if pdf_1D is from the posterior)
    est_ind = np.argmax(pdf_1D)
    out_dict["Estimate"] = val_arr[est_ind]
    
    # Generate cumulative density function (CDF) using trapezoidal integration:
    cdf_1D = cumtrapz(pdf_1D, x=val_arr, initial=0)
    # initial=0 => cdf_1D has same length as pdf_1D; first CDF entry will be 0;
    # the last will be 1 (once normalised).
    # Normalise CDF (should be very close to normalised already):
    cdf_1D /= cdf_1D[-1]

    # Calculate credible intervals
    for CI in [68, 95]:
        # Find the percentiles of lower and upper bounds of CI:
        lower_prop = (1.0 - CI/100.0) / 2.0 # Lower percentile as proportion
        upper_prop = 1.0 - lower_prop       # Upper percentile as proportion
        
        # Find value corresponding to the lower bound
        lower_ind_arr = index_array[cdf_1D < lower_prop]
        if lower_ind_arr.size == 1:
            # The 1st CDF value (0) is the only value below the lower bound
            lower_val = -np.inf # Indicate that we don't have a lower bound
        else: # Use the first value below the lower bound, to be conservative
            lower_val = val_arr[ lower_ind_arr[-1] ]

        # Find value corresponding to the upper bound
        upper_ind_arr = index_array[cdf_1D > upper_prop]
        if upper_ind_arr.size == 1:
            # The last CDF value (1) is the only value above the upper bound
            upper_val = np.inf # Indicate that we don't have an upper bound
        else: # Use the first value above the upper bound, to be conservative
            upper_val = val_arr[ upper_ind_arr[0] ]

        # Save results into output dictionary:
        CI_code = "CI" + str(CI)  # e.g. "CI68"
        out_dict[CI_code+"_low"] = lower_val
        out_dict[CI_code+"_high"] = upper_val

        # Check best estimate is actually inside the CI:
        is_in_CI = (lower_val <= out_dict["Estimate"] <= upper_val)
        out_dict["Est_in_"+CI_code+"?"] = bool_map[is_in_CI]

    # Count local maxima
    tup_max_inds = argrelextrema( pdf_1D, np.greater )
    out_dict["n_local_maxima"] = tup_max_inds[0].size
    # argrelextrema doesn't pick up maxima on the ends, so check ends:
    if pdf_1D[0] > pdf_1D[1]:
        out_dict["n_local_maxima"] += 1
    if pdf_1D[-1] > pdf_1D[-2]:
        out_dict["n_local_maxima"] += 1

    # Check if the mass of probabiltiy is at an edge of the parameter space:
    # Lower bound: Check 3rd value of CDF (the 1st is 0 by design)
    out_dict["P(lower)"] = cdf_1D[2]
    out_dict["P(lower)>50%?"] = bool_map[ (out_dict["P(lower)"] > 0.5) ]
    out_dict["Est_at_lower?"] = bool_map[ (est_ind <= 2) ]
    # Upper bound: Check 3rd-last value of CDF (the last is 1 by design)
    out_dict["P(upper)"] = 1.0 - cdf_1D[-3]
    out_dict["P(upper)>50%?"] = bool_map[ (out_dict["P(upper)"] > 0.5) ]
    out_dict["Est_at_upper?"] = bool_map[ (est_ind >= cdf_1D.size - 4) ]
    
    return out_dict



class NB_Result(object):
    """
    Class to hold the NebulaBayes results including the likelihood, prior and
    posterior, marginalised PDFs and parameter estimates.
    An instance of this class is returned to the user each time NebulaBayes is
    run.
    """
    def __init__(self, Interpd_grids, DF_obs, ND_PDF_Plotter_1, norm_line,
                                          deredden, input_prior, line_plot_dir):
        """
        Initialise an instance of the class and perform Bayesian parameter
        estimation.
        """
        self.DF_obs = DF_obs # Store for user
        self.Plotter = ND_PDF_Plotter_1 # To plot ND PDFs
        self.deredden = deredden # Boolean - dereddeden obs fluxes over whole grid?
        self._line_plot_dir = line_plot_dir
        Grid_spec = Grid_description(Interpd_grids.param_names,
                           list(Interpd_grids.paramName2paramValueArr.values()))
        self.Grid_spec = Grid_spec

        # Calculate arrays of observed fluxes over the grid (possibly dereddening)
        self._make_obs_flux_arrays(Interpd_grids, norm_line)

        # Calculate the likelihood over the grid:
        print("Calculating likelihood...")
        raw_likelihood = self._calculate_likelihood(Interpd_grids, norm_line)
        self.Likelihood = NB_nd_pdf(raw_likelihood, self, Interpd_grids, DF_obs)

        # Calculate the prior over the grid:
        print("Calculating prior...")
        raw_prior = calculate_prior(input_prior, DF_obs, Interpd_grids.grids,
                                                   Interpd_grids.grid_rel_error)
        self.Prior = NB_nd_pdf(raw_prior, self, Interpd_grids, DF_obs)

        # Calculate the posterior using Bayes' Theorem:
        # (note that the prior and likelihood pdfs are now normalised)
        print("Calculating posterior using Bayes' Theorem...")
        raw_posterior = self.Likelihood.nd_pdf * self.Prior.nd_pdf # Bayes' theorem
        self.Posterior = NB_nd_pdf(raw_posterior, self, Interpd_grids, DF_obs)



    def _make_obs_flux_arrays(self, Interpd_grids, norm_line):
        """
        Make observed flux arrays covering the entire grid.
        If requested by the user, the observed fluxes are dereddened to match
        the Balmer decrement everywhere in the grid.  Otherwise the flux value
        for a line is uniform over the n-D grid.
        The observed fluxes have already been normalised to norm_line.
        """
        DF_obs = self.DF_obs
        if self.deredden:
            # Deredden observed fluxes at every interpolated gridpoint to match
            # the model Balmer decrement at that gridpoint.
            if norm_line != "Hbeta":
                raise ValueError("Dereddening is only supported for "
                                 "norm_line = 'Hbeta'")
            # Array of Balmer decrements across the grid:
            grid_BD_arr = Interpd_grids.grids["Halpha"] / Interpd_grids.grids["Hbeta"]
            # (Grid fluxes may not have been normalised to Hbeta == 1)
            if not np.all(np.isfinite(grid_BD_arr)): # Sanity check
                raise ValueError("Something went wrong - the array of predicted"
                                 " Balmer decrements over the grid contains a"
                                 " non-finite value!")
            if not np.all(grid_BD_arr > 0): # Sanity check
                raise ValueError("Something went wrong - the array of predicted"
                                 " Balmer decrements over the grid contains at"
                                 " least one non-positive value!")

            obs_flux_arrs, obs_flux_err_arrs = do_dereddening(
                        DF_obs["Wavelength"].values, DF_obs["Flux"].values,
                        DF_obs["Flux_err"].values, BD=grid_BD_arr, normalise=True)
            # The output fluxes and errors are normalised to Hbeta == 1.
        
        else: # Use the input observed fluxes, which hopefully have already
              # been dereddened if necessary.
            shape = Interpd_grids.shape
            obs_flux_arrs = [np.full(shape, f) for f in DF_obs["Flux"].values]
            obs_flux_err_arrs = [np.full(shape, e) for e in DF_obs["Flux_err"].values]

        # Now obs_flux_arrs is a list of arrays corresponding to the list
        # of observed fluxes, where each array has the same shape as the 
        # model grid.  The list obs_flux_err_arrs is the same, but for errors.
        self.obs_flux_arrs = obs_flux_arrs
        self.obs_flux_err_arrs = obs_flux_err_arrs



    def _calculate_likelihood(self, Interpd_grids, norm_line):
        """
        Calculate the (linear) likelihood over the entire N-D grid at once.
        The likelihood is not yet normalised - that will be done later.
        The emission line grids have been interpolated prior to being inputted
        into this method.  The likelihood is a product of PDFs, one for each
        contributing emission line.  We save out a plot of the PDF for each line
        if the uses wishes.
        """
        # Systematic relative error in normalised grid fluxes as a linear
        # proportion:
        pred_flux_rel_err = Interpd_grids.grid_rel_error

        # Arrays of observed fluxes over the whole grid (may have been
        # dereddened at every point in the grid)
        obs_flux_arrs = self.obs_flux_arrs
        obs_flux_err_arrs = self.obs_flux_err_arrs

        # Normalise the interpolated grid fluxes if necessary
        if Interpd_grids.norm_line != norm_line:
            norm_grid = Interpd_grids.grids[norm_line].copy()
            # Copy array so it won't become all "1.0" in the middle of normalising!
            bad = (norm_grid == 0) # For when we divide by norm_grid
            for line in Interpd_grids.grids:
                Interpd_grids.grids[line] /= norm_grid  # In-place division
                # Replace any NaNs we produced by dividing by zero:
                Interpd_grids.grids[line][bad] = 0
            Interpd_grids.norm_line = norm_line # Store for later reference

        # Initialise log likelihood with 0 everywhere
        log_likelihood = np.zeros(Interpd_grids.shape, dtype="float")
        # # Initialise linear likelihood with 1 everywhere
        # likelihood = np.ones(Interpd_grids.shape, dtype="float")
        for i, line in enumerate(self.DF_obs.index):
            pred_flux_i = Interpd_grids.grids[line]
            obs_flux_i = obs_flux_arrs[i]
            obs_flux_err_i = obs_flux_err_arrs[i]
            assert obs_flux_i.shape == pred_flux_i.shape
            
            # Use a log version of equation 3 on pg 4 of Blanc et al. 2015 (IZI)
            # N.B. var is the sum of variances due to both the measured and
            # modelled fluxes
            var = obs_flux_err_i**2 + (pred_flux_rel_err * pred_flux_i)**2
            line_contribution = ( - (( obs_flux_i - pred_flux_i )**2 / 
                                     ( 2.0 * var ))  -  0.5*np.log( var ) )
            log_likelihood += line_contribution
            # N.B. "log" is base e
            # # Linear version:
            # line_contribution = ((1./np.sqrt(var)) *
            #         np.exp( -(( obs_flux_i - pred_flux_i )**2 / ( 2.0 * var )))  
            # log_likelihood *= line_contribution

            # Plot the ND PDF for each line if requested:
            if self._line_plot_dir is not None:
                # Assume we're using the log version of the likelihood equation
                line_pdf = np.exp(line_contribution - line_contribution.max())
                outname = os.path.join(self._line_plot_dir,
                                    line + "_PDF_contributes_to_likelihood.pdf")
                Line_PDF = NB_nd_pdf(line_pdf, self, Interpd_grids)
                Line_PDF.Grid_spec.param_display_names = Interpd_grids.param_display_names
                print("Plotting pdf for line {0}...".format(line))
                self.Plotter(Line_PDF, outname)

        # log_likelihood += np.log(2 * np.pi)
        log_likelihood -= log_likelihood.max() # Ensure max is 0 (log space); 1 (linear)
        return np.exp(log_likelihood) # The linear likelihood N-D array   


