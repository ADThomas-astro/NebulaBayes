from __future__ import print_function, division
from collections import OrderedDict as OD
import itertools
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Some work to ensure we're importing this version of NebulaBayes:
THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
NB_PARENT_DIR = os.path.join(os.path.split(THIS_FILE_DIR)[0], "src")
sys.path.insert(1, NB_PARENT_DIR)
from NebulaBayes import NB_Model, __version__
from NebulaBayes.src.NB1_Process_grids import RegularGridResampler



"""
Test suite to test NebulaBayes.  Mostly functional and regression tests, with
some unit tests as well.

Works with Python 2 and Python 3.

Adam D. Thomas 2017
"""

clean_up = True  # Delete test output files after running?
test_dir = os.path.join(THIS_FILE_DIR, "test_outputs")  # For outputs





###############################################################################
# Helper functions
def build_grid(param_range_dict, line_peaks_dict, n_gridpts_list, std_frac=0.25):
    """
    Initialise a grid - create a pandas DataFrame table.  Fluxes for each
    emission line form a Gaussian ball around a specified point.
    param_range_dict: Ordered dict mapping parameter names to a tuple giving
                      the parameter minimum and maximum
    line_peaks_dict: Ordered dict mapping line names to the location
                     (as a tuple) of the peak of the line flux in the grid, in
                     gridpoint index coordinates (from zero)
    std_frac:  Fraction of the range in each dimension used for the std
    n_gridpts_list is a list of the number of gridpoints in each dimension.
    """
    param_names = list(param_range_dict.keys())
    param_val_arrs = [np.linspace(r[0], r[1], n) for r,n in zip(
                                    param_range_dict.values(), n_gridpts_list)]
    line_names = list(line_peaks_dict.keys())
    std = np.array([(r[1] - r[0]) * std_frac for r in param_range_dict.values()])

    line_peak_vals = {}
    for line, peak_inds in line_peaks_dict.items():
        line_peak = []
        for p, peak_ind, val_arr in zip(param_names, peak_inds, param_val_arrs):
            p_min, dp = val_arr[0], np.diff(val_arr)[0]
            line_peak.append(p_min + peak_ind*dp)
        line_peak_vals[line] = line_peak  # An ND list corresponding to peak_inds

    flux_fns = {}
    for l,peak_tuple in line_peaks_dict.items():
        peak = np.array(line_peak_vals[l])  # ND vector
        def gaussian(x):
            # N.B. x, peak and std are all ND vectors
            distance = np.sqrt(np.sum(((x - peak) / std)**2))
            return np.exp(-distance / 2)
        flux_fns[l] = gaussian

    # Make DataFrame table:
    columns = param_names + line_names
    n_gridpts = np.product(n_gridpts_list)
    OD_for_DF = OD([(c, np.full(n_gridpts, np.nan)) for c in columns])
    DF_grid = pd.DataFrame(OD_for_DF)

    # Iterate over rows, filling in the table
    for i, p_tuple in enumerate(itertools.product(*param_val_arrs)):
        # Add parameter values into their columns:
        for p,n in zip(p_tuple, param_names):
            DF_grid.loc[i,n] = p
        # Add "model" line fluxes into their columns:
        for l in line_names:
            DF_grid.loc[i,l] = flux_fns[l](np.array(p_tuple))

    return DF_grid



def extract_grid_fluxes_i(DF, p_name_ind_map, line_names):
    """
    Extract emission line fluxes from a grid (represented as a DataFrame) by
    inputting gridpoint indices and taking the fluxes at the nearest gridpoint
    """
    val_arrs = {p:np.unique(DF[p].values) for p in p_name_ind_map}
    assert len(DF) == np.product([len(v) for v in val_arrs.values()])
    where = np.full(len(DF), 1, dtype=bool)
    for p,ind in p_name_ind_map.items():
        where &= (DF.loc[:,p] == val_arrs[p][ind])
    assert np.sum(where) == 1

    return [DF[line].values[where][0] for line in line_names]



###############################################################################

# Helper class
class Base_2D_Grid_2_Lines(unittest.TestCase):
    """
    Base class holding setup and cleanup methods to make a 2D grid with only 2
    emission lines, and using a 2D Gaussian to make the grid.  There are only
    two lines, but one has fluxes set to all 1 and is just for normalisation.
    """
    params = ["p1", "p2"]
    param_range_dict = OD( [("p1", (-5, 3)), ("p2", (1.2e6, 15e6))] )
    n_gridpts_list = (11, 9) # Number of gridpoints in each dimension
    interpd_shape = (50, 45)
    lines = ["L1", "L2"] # Line names
    line_peaks = [8, 5]  # Gridpoint indices from zero

    @classmethod
    def setUpClass(cls):
        """ Make grid and run NebulaBayes to obtain the result object """
        line_peaks_dict = OD([(l,cls.line_peaks) for l in cls.lines])
        cls.DF = build_grid(cls.param_range_dict, line_peaks_dict, cls.n_gridpts_list)
        cls.val_arrs = OD([(p,np.unique(cls.DF[p].values)) for p in cls.params])
        cls.DF.loc[:,"L1"] = 1.  # We'll normalise by this line
        cls.grid_file = os.path.join(test_dir, cls.__name__ + "_grid.csv")
        cls.DF.to_csv(cls.grid_file, index=False)

        cls.NB_Model_1 = NB_Model(cls.grid_file, cls.params, cls.lines,
                                           interpd_grid_shape=cls.interpd_shape)

    @classmethod
    def tearDownClass(cls):
        """ Remove the output when tests in this class have finished """
        if clean_up:
            os.remove(cls.grid_file)
            if hasattr(cls, "posterior_plot"):
                os.remove(cls.posterior_plot)



###############################################################################

class Test_Obs_from_Peak_Gridpoint_2D_Grid_2_Lines(Base_2D_Grid_2_Lines):
    """
    Test for a grid from Base_2D_Grid_2_Lines:  Take a gridpoint that is at
    the peak of the Gaussian ball of emission line fluxes, and check that
    treating these fluxes as observations leads to correct estimates from
    NebulaBayes.
    """
    test_gridpoint = [8, 5]  # From zero.  [11, 9] total gridpoints in each dim

    @classmethod
    def setUpClass(cls):
        super(Test_Obs_from_Peak_Gridpoint_2D_Grid_2_Lines, cls).setUpClass()
        test_pt = OD(zip(cls.params, cls.test_gridpoint)) # Map params to gridpt indices
        obs_fluxes = extract_grid_fluxes_i(cls.DF, test_pt, ["L1", "L2"])
        obs_errors = [f / 7. for f in obs_fluxes]

        cls.posterior_plot = os.path.join(test_dir, cls.__name__ + "_posterior.pdf")
        kwargs = {"posterior_plot": cls.posterior_plot,
                  "line_plot_dir": test_dir, "norm_line": "L1"}

        cls.Result = cls.NB_Model_1(obs_fluxes, obs_errors, cls.lines, **kwargs)

    def test_output_deredden_flag(self):
        self.assertTrue(self.Result.deredden is False)

    def test_parameter_estimates(self):
        """ Ensure the parameter estimates are as expected """
        DF_est = self.Result.Posterior.DF_estimates
        self.assertTrue(all(p in DF_est.index for p in self.params))
        # Tolerance for distance between gridpoint we chose and the estimate:
        grid_sep_frac = 0.1  # Allowed fraction of distance between gridpoints
        for p, test_ind in zip(self.params, self.test_gridpoint):
            tol = np.diff(self.val_arrs[p])[0] * grid_sep_frac
            value = self.val_arrs[p][test_ind]  # Expected parameter value
            est = DF_est.loc[p, "Estimate"]  # NebulaBayes estimate
            self.assertTrue(np.isclose(est, value, atol=tol))

    def test_raw_Grid_spec(self):
        """ Ensure the raw grid spec is as expected """
        RGrid_spec = self.NB_Model_1.Raw_grids
        self.assertEqual(RGrid_spec.param_names, self.params)
        self.assertEqual(RGrid_spec.ndim, len(self.params))
        self.assertEqual(RGrid_spec.shape, self.n_gridpts_list)
        self.assertEqual(RGrid_spec.n_gridpoints, np.product(self.n_gridpts_list))
        for a1, a2 in zip(RGrid_spec.param_values_arrs, self.val_arrs.values()):
            self.assertTrue(np.allclose(np.asarray(a1), np.asarray(a2)))

    def test_interpolated_Grid_spec(self):
        """ Ensure the interpolated grid spec is as expected """
        IGrid_spec = self.Result.Grid_spec
        self.assertEqual(IGrid_spec.param_names, self.params)
        self.assertEqual(IGrid_spec.param_display_names, self.params)
        self.assertEqual(IGrid_spec.shape, tuple(self.interpd_shape))
        self.assertEqual(IGrid_spec.n_gridpoints, np.product(self.interpd_shape))



###############################################################################

class Test_Obs_from_nonPeak_Gridpoint_2D_Grid_2_Lines(Base_2D_Grid_2_Lines):
    """
    Test for a grid from Base_2D_Grid_2_Lines:  Take a gridpoint that is NOT at
    the peak of the Gaussian ball of emission line fluxes.
    Note that we don't check the values in the posterior or parameter
    estimates - there isn't an obvious way to do this here.
    We also test that a numpy array prior is accepted.
    """
    longMessage = True  # Append messages to existing message
    test_gridpoint = [6, 4]  # From zero.  [11, 9] total gridpoints in each dim,
                             # the line peak is at line_peaks = [8, 5]

    @classmethod
    def setUpClass(cls):
        super(Test_Obs_from_nonPeak_Gridpoint_2D_Grid_2_Lines, cls).setUpClass()
        test_pt = OD(zip(cls.params, cls.test_gridpoint)) # Map params to gridpt indices
        obs_fluxes = extract_grid_fluxes_i(cls.DF, test_pt, ["L1", "L2"])
        obs_errors = [f / 7. for f in obs_fluxes]

        cls.posterior_plot = os.path.join(test_dir, cls.__name__ + "_posterior.pdf")
        kwargs = {"posterior_plot": cls.posterior_plot, "norm_line": "L1",
                  "prior": np.ones(cls.NB_Model_1.Interpd_grids.shape)
                 }

        cls.Result = cls.NB_Model_1(obs_fluxes, obs_errors, cls.lines, **kwargs)

    def test_output_deredden_flag(self):
        self.assertTrue(self.Result.deredden is False)

    def test_output_chi2_positive(self):
        chi2 = self.Result.Posterior.best_model["chi2"]
        self.assertTrue(chi2 > 0, msg="chi2 is " + str(chi2))

    def test_output_extinction_is_NA(self):  # Since we didn't deredden
        Av = self.Result.Posterior.best_model["extinction_Av_mag"]
        self.assertTrue(Av == "NA (deredden is False)")

    def test_parameters_in_output(self):
        """ Check all parameters are found in output """
        DF_est = self.Result.Posterior.DF_estimates
        self.assertTrue(all(p in DF_est.index for p in self.params))
        # Posterior is shaped like a donut.  Check for a single local min?



###############################################################################
# Test the NebulaBayes ND linear interpolation

class test_linear_interpolation_1D(unittest.TestCase):
    def test_linear_interpolation_simple_1D(self):
        R1 = RegularGridResampler([[10, 20, 30]], [5])
        R1_pout, R1_arr = R1(np.array([-100,-200,-300]))
        assert np.array_equal(R1_pout[0], np.array([ 10.,  15.,  20.,  25.,  30.]))
        assert np.array_equal(R1_arr, np.array([-100., -150., -200., -250., -300.]))


class test_linear_interpolation_2D(unittest.TestCase):
    def test_linear_interpolation_simple_2D(self):
        R2 = RegularGridResampler([[10, 20, 30], [8e4, 9e4]], [5,2])
        R2_pout, R2_arr = R2(np.array([[-100,-2000],
                                       [-300,-4000],
                                       [-500,-6000]]))
        assert np.array_equal(R2_pout[0], np.array([ 10.,  15.,  20.,  25.,  30.]))
        assert np.array_equal(R2_pout[1], np.array([8e4, 9e4]))
        assert np.array_equal(R2_arr, np.array([[-100., -2000.],
                                                [-200., -3000.],
                                                [-300., -4000.],
                                                [-400., -5000.],
                                                [-500., -6000.]]))

class test_linear_interpolation_3D(unittest.TestCase):
    def test_linear_interpolation_simple_3D(self):
        R3 = RegularGridResampler([[-2, -1],[10, 20, 30], [8e4, 9e4]], [2, 5, 2])
        R3_pout, R3_arr = R3(np.array([[[-100, -2000],
                                        [-300, -4000],
                                        [-500, -6000]],
                                       [[-100, -2000],
                                        [-300, -4000],
                                        [-500, -6000]] ]))
        assert np.array_equal(R3_pout[0], np.array([-2, -1]))
        assert np.array_equal(R3_pout[1], np.array([10.,  15.,  20.,  25.,  30.]))
        assert np.array_equal(R3_pout[2], np.array([8e4, 9e4]))
        assert np.array_equal(R3_arr, np.array([[[-100., -2000.],
                                                 [-200., -3000.],
                                                 [-300., -4000.],
                                                 [-400., -5000.],
                                                 [-500., -6000.]],
                                                [[-100., -2000.],
                                                 [-200., -3000.],
                                                 [-300., -4000.],
                                                 [-400., -5000.],
                                                 [-500., -6000.]] ]))



###############################################################################

class Test_1D_grid_and_public_attributes(unittest.TestCase):
    """
    Test that a 1D grid works and gives expected results.
    We use a gaussian 1D "grid", and input a point at the peak into NB to
    ensure NB finds the correct point.
    We also test that a DataFrame grid table is accepted.
    """
    longMessage = True  # Append messages to existing message

    @classmethod
    def setUpClass(cls):
        # Make a 1D grid:
        test_gridpoint = 45  # From zero
        cls.test_gridpoint = test_gridpoint
        p_vals = np.linspace(-2, 8, 100)
        cls.p_vals = p_vals
        peak1, peak2, peak3 = p_vals[8], p_vals[60], p_vals[83]
        std1, std2, std3 = 1.1, 1.8, 4.3
        flux_0 = 3.0 * np.ones_like(p_vals)
        flux_1 = 13. * np.exp(-np.sqrt(((p_vals - peak1) / std1)**2) / 2)
        flux_2 = 13. * np.exp(-np.sqrt(((p_vals - peak2) / std2)**2) / 2)
        flux_3 = 21. * np.exp(-np.sqrt(((p_vals - peak3) / std3)**2) / 2)
        cls.lines = ["l0", "l1", "l2", "l3"]
        DF_grid1D = pd.DataFrame({"P0":p_vals, "l0":flux_0, "l1":flux_1,
                                  "l2":flux_2, "l3":flux_3})
        obs_fluxes = [x[test_gridpoint] for x in [flux_0,flux_1,flux_2,flux_3]]
        obs_errors = [f / 7. for f in obs_fluxes]

        cls.posterior_plot = os.path.join(test_dir,
                                          cls.__name__ + "_posterior.pdf")
        cls.best_model_table = os.path.join(test_dir,
                                          cls.__name__ + "_best_model.csv")
        cls.NB_Model_1 = NB_Model(DF_grid1D, ["P0"], cls.lines,
                                  interpd_grid_shape=[300])
        # We test the case-insensitivity of the norm_line, by writing
        # "L0" instead of "l0" here:
        kwargs = {"posterior_plot":cls.posterior_plot, "norm_line":"L0",
                  "best_model_table":cls.best_model_table}
        cls.Result = cls.NB_Model_1(obs_fluxes, obs_errors, cls.lines, **kwargs)

    def test_output_deredden_flag(self):
        self.assertTrue(self.Result.deredden is False)

    def test_output_chi2_positive(self):
        chi2 = self.Result.Posterior.best_model["chi2"]
        self.assertTrue(chi2 > 0, msg="chi2 is " + str(chi2))

    def test_output_extinction_is_NA(self):  # Since we didn't deredden
        Av = self.Result.Posterior.best_model["extinction_Av_mag"]
        self.assertTrue(Av == "NA (deredden is False)")

    def test_parameter_estimate(self):
        """ Ensure the single parameter estimate is as expected """
        DF_est = self.Result.Posterior.DF_estimates
        self.assertTrue("P0" in DF_est.index)
        lower = self.p_vals[self.test_gridpoint - 1]
        upper = self.p_vals[self.test_gridpoint + 1]
        est = DF_est.loc["P0", "Estimate"]
        self.assertTrue(lower < est < upper, msg="{0}, {1}, {2}".format(
                                                            lower, est, upper))

    def test_NB_Model_attributes(self):
        """ Check that the list of public attributes is what is documented """
        public_attrs = sorted([a for a in dir(self.NB_Model_1)
                                                    if not a.startswith("_")])
        expected_attrs = ["Interpd_grids", "Raw_grids"]
        self.assertTrue(public_attrs == expected_attrs, msg=str(public_attrs))

    def test_NB_Result_attributes(self):
        """ Check that the list of public attributes is what is documented """
        public_attrs = sorted([a for a in dir(self.Result)
                                                    if not a.startswith("_")])
        expected_attrs = ["DF_obs", "Grid_spec", "Likelihood", "Plot_Config",
                          "Plotter", "Posterior", "Prior", "deredden",
                          "obs_flux_arrs", "obs_flux_err_arrs"]
        self.assertTrue(public_attrs == expected_attrs, msg=str(public_attrs))

    def test_NB_nd_pdf_attributes(self):
        """ Check that the list of public attributes is what is documented """
        public_attrs = sorted([a for a in dir(self.Result.Posterior)
                                                    if not a.startswith("_")])
        expected_attrs = sorted(["DF_estimates", "Grid_spec", "best_model",
                        "marginalised_1D", "marginalised_2D", "name", "nd_pdf"])
        self.assertTrue(public_attrs == expected_attrs, msg=str(public_attrs))

    @classmethod
    def tearDownClass(cls):
        """ Remove the output when tests in this class have finished """
        if clean_up:
            if hasattr(cls, "posterior_plot"):
                os.remove(cls.posterior_plot)
            if hasattr(cls, "best_model_table"):
                os.remove(cls.best_model_table)



###############################################################################

class Test_default_initialisation(unittest.TestCase):
    """
    Test that we can initialise fully default HII and NLR NB models
    """
    def test_default_HII_initialisation(self):
        NB_Model("HII")

    def test_default_NLR_initialisation(self):
        NB_Model("NLR")



###############################################################################

class Test_real_data_with_dereddening(unittest.TestCase):
    """
    Test some real data, from the S7 nuclear spectrum for NGC4691, a star-
    forming galaxy.  Include a line ratio prior and dereddening in NebulaBayes.
    Test saving plots for all 3 Bayes Theorem PDFs.
    """
    longMessage = True  # Append messages to existing message

    lines = ["OII3726_29", "Hgamma", "OIII4363", "Hbeta", "OIII5007",
                 "NI5200", "OI6300", "Halpha", "NII6583", "SII6716", "SII6731"]
    obs_fluxes = [1.22496, 0.3991, 0.00298, 1.0, 0.44942,
                  0.00766, 0.02923, 4.25103, 1.65312, 0.45598, 0.41482]
    obs_errs = [0.00303, 0.00142, 0.00078, 0.0017, 0.0012,
                0.00059, 0.00052, 0.00268, 0.00173, 0.00102, 0.00099]
    obs_wavelengths = [3727.3, 4340.5, 4363.2, 4861.3, 5006.8,
                       5200.3, 6300.3, 6562.8, 6583.2, 6716.4, 6730.8]

    @classmethod
    def setUpClass(cls):
        cls.prior_plot = os.path.join(test_dir,
                                          cls.__name__ + "_prior.pdf")
        cls.likelihood_plot = os.path.join(test_dir,
                                          cls.__name__ + "_likelihood.pdf")
        cls.posterior_plot = os.path.join(test_dir,
                                          cls.__name__ + "_posterior.pdf")
        cls.estimate_table = os.path.join(test_dir,
                                    cls.__name__ + "_parameter_estimates.csv")
        # Test different values along each dimension in interpd_grid_shape
        cls.NB_Model_1 = NB_Model("HII", grid_params=None, lines_list=cls.lines,
                                  interpd_grid_shape=[100, 130, 80])

        kwargs = {"prior_plot": cls.prior_plot,
                  "likelihood_plot": cls.likelihood_plot,
                  "posterior_plot": cls.posterior_plot,
                  "estimate_table": cls.estimate_table,
                  "deredden": True, "obs_wavelengths": cls.obs_wavelengths,
                  "prior":[("SII6716","SII6731")],
                  "plot_configs": [{"table_on_plot": True,
                                    "legend_fontsize": 5}]*4,
                  }
        cls.Result = cls.NB_Model_1(cls.obs_fluxes, cls.obs_errs, cls.lines,
                                    **kwargs)

    def test_parameter_estimates(self):
        """
        Regression check on parameter estimates.
        """
        ests = self.Result.Posterior.DF_estimates["Estimate"]  # pandas Series
        self.assertTrue(np.isclose(ests["12 + log O/H"], 8.73615, atol=0.0001),
                        msg=str(ests["12 + log O/H"]))
        self.assertTrue(np.isclose(ests["log P/k"], 6.82636, atol=0.0001),
                        msg=str(ests["log P/k"]))
        self.assertTrue(np.isclose(ests["log U"], -2.84848, atol=0.0001),
                        msg=str(ests["log U"]))

    def test_estimate_bounds_checks(self):
        """
        Ensure that the "checking columns" in the estimate table are all
        showing that the estimates are good.
        """
        DF = self.Result.Posterior.DF_estimates  # Parameter estimate table
        for p in ["12 + log O/H", "log P/k", "log U"]:
            for col in ["Est_in_CI68?", "Est_in_CI95?"]:
                self.assertTrue(DF.loc[p,col] == "Y")
            for col in ["Est_at_lower?", "Est_at_upper?", "P(lower)>50%?",
                        "P(upper)>50%?"]:
                self.assertTrue(DF.loc[p,col] == "N")
            self.assertTrue(DF.loc[p,"n_local_maxima"] == 1)

    def test_chi2(self):
        """
        Regression check that chi2 doesn't change
        """
        chi2 = self.Result.Posterior.best_model["chi2"]
        self.assertTrue(np.isclose(chi2, 1401.3, atol=0.2))

    def test_all_zero_prior(self):
        """
        We permit an all-zero prior - check that it works (a warning should
        be printed).
        """
        shape = self.NB_Model_1.Interpd_grids.shape
        self.Result1 = self.NB_Model_1(self.obs_fluxes, self.obs_errs,
                                       self.lines, prior=np.zeros(shape))


    @classmethod
    def tearDownClass(cls):
        """ Remove the output files when tests in this class have finished """
        if clean_up:
            files = [cls.prior_plot, cls.likelihood_plot, cls.posterior_plot,
                     cls.estimate_table]
            for file_i in files:
                os.remove(file_i)



###############################################################################

class Test_upper_bounds_1D(unittest.TestCase):
    """
    Test the treatment of upper bounds.  We use a 1D grid.
    """
    longMessage = True  # Append messages to existing message

    lines       = ["line1", "line2", "line3", "line4", "line5", "line6"]
    obs_fluxes  = [    1.0,     8.0,    10.2, -np.inf, -np.inf, -np.inf]
    obs_errs    = [   0.05,     0.3,     3.1,     0.3,     0.4,     0.2]
    pred_fluxes = [    1.0,     5.0,    10.2,     0.1,     0.4,     0.4]
    # The pred_fluxes are at the "peak" of the grid, that we'll input to NB.

    @classmethod
    def setUpClass(cls):
        n = 100  # Length of grid
        best_i = 65
        DF_grid1D = pd.DataFrame()
        DF_grid1D["p0"] = np.arange(n) - 572.3  # Linear
        DF_grid1D["dummy"] = np.exp(-((DF_grid1D["p0"] -
                                      DF_grid1D["p0"].values[best_i])/17.2)**2)
        DF_grid1D["dummy"] = DF_grid1D["dummy"].values / DF_grid1D["dummy"].max()
        for line, pred_flux in zip(cls.lines, cls.pred_fluxes):
            DF_grid1D[line] = DF_grid1D["dummy"].values * pred_flux
            # All of the fluxes peak at the point we'll input to NB
        DF_grid1D["line1"] = np.ones_like(DF_grid1D["line1"].values)
        cls.expected_p0 = DF_grid1D["p0"].values[best_i]

        # Note that we set grid_error to zero!
        cls.NB_Model_1 = NB_Model(DF_grid1D, grid_params=["p0"], grid_error=0,
                                lines_list=cls.lines, interpd_grid_shape=[500])
        kwargs = {"deredden": False, "norm_line": "line1",
                  "line_plot_dir": test_dir}
        cls.Result = cls.NB_Model_1(cls.obs_fluxes, cls.obs_errs, cls.lines,
                                    **kwargs)

    def test_parameter_estimates(self):
        """
        Regression test - check the parameter estimate is as expected.
        """
        DF_est = self.Result.Posterior.DF_estimates  # DataFrame
        p0_est = DF_est.loc["p0", "Estimate"]
        self.assertTrue(np.isclose(p0_est, self.expected_p0, atol=1))

    @classmethod
    def tearDownClass(cls):
        """ Remove the output files when tests in this class have finished """
        if clean_up:
            files = [os.path.join(test_dir, l +
                     "_PDF_contributes_to_likelihood.pdf") for l in cls.lines]
            for file_i in files:
                os.remove(file_i)



###############################################################################

class Test_all_zero_likelihood(unittest.TestCase):
    """
    Test forcing a log_likelihood of all -inf, so the likelihood is all zero.
    """
    longMessage = True  # Append messages to existing message

    lines       = ["Halpha", "Hbeta", "OIII4363", "OIII5007", "NII6583"]
    obs_fluxes  = [   1e250,       1,    1.2e250,    1.2e250,     1e250]
    obs_errs    = [   0.004,       1,      0.005,      0.003,     0.002]

    @classmethod
    def setUpClass(cls):
        cls.NB_Model_1 = NB_Model("HII", lines_list=cls.lines, grid_error=0.01,
                                  interpd_grid_shape=[30,30,30])
        kwargs = {"deredden": False, "norm_line": "Hbeta",
                  "prior":[("NII6583","Halpha")]}
        cls.Result = cls.NB_Model_1(cls.obs_fluxes, cls.obs_errs, cls.lines,
                                    **kwargs)

    def test_likelihood_all_zero(self):
        """
        Regression test - check likelihood is all zero.
        """
        likelihood = self.Result.Likelihood.nd_pdf
        self.assertTrue(np.all(likelihood == 0))

    def test_posterior_all_zero(self):
        """
        Regression test - check posterior is all zero.
        """
        posterior = self.Result.Posterior.nd_pdf
        self.assertTrue(np.all(posterior == 0))



###############################################################################

class Test_data_that_matches_models_poorly(unittest.TestCase):
    """
    Test inputting fluxes and errors that are very poorly fit by the entire
    model grid.  In this case most of the likelihood is zero, and using a
    reasonable-ish prior gives a posterior that is zero everywhere.
    """
    longMessage = True  # Append messages to existing message

    lines       = ["Halpha", "Hbeta", "OIII4363", "OIII5007", "NII6583"]
    obs_fluxes  = [     3.1,       1,        1.8,        5.1,       1.2]
    obs_errs    = [    0.01,       1,       0.01,       0.01,      0.01]
    # Note the very small errors

    @classmethod
    def setUpClass(cls):
        cls.NB_Model_1 = NB_Model("HII", lines_list=cls.lines, grid_error=0.01,
                                  interpd_grid_shape=[30,30,30])
        kwargs = {"deredden": False, "norm_line": "Hbeta",
                  "prior":[("NII6583", "OIII4363")]}
        cls.Result = cls.NB_Model_1(cls.obs_fluxes, cls.obs_errs, cls.lines,
                                    **kwargs)

    def test_likelihood_mostly_zero(self):
        """
        Regression test - check likelihood is mostly zero.
        """
        likelihood = self.Result.Likelihood.nd_pdf
        self.assertTrue(np.sum(likelihood != 0) < 65)

    def test_posterior_all_zero(self):
        """
        Regression test - check posterior is all zero.
        """
        posterior = self.Result.Posterior.nd_pdf
        self.assertTrue(np.all(posterior == 0))



###############################################################################

class Test_raising_errors(unittest.TestCase):
    """
    Test raising errors on bad inputs
    """

    @classmethod
    def setUpClass(cls):
        try:
            # Python 3
            cls.assertRaisesRE = cls.assertRaisesRegex
        except AttributeError:
            # Python 2
            cls.assertRaisesRE = cls.assertRaisesRegexp

    def test_bad_grid_parameter_with_too_few_unique_values(self):
        """
        Test correct error is raised if there are too few unique values for
        a grid parameter.
        """
        DF = pd.DataFrame({"p1": [4, 4, 4, 4], "p2": [1, 2, 3, 4],
                           "l2": [5, 6, 7, 8]})
        self.assertRaisesRE(ValueError, "3 unique values are required",
                            NB_Model, DF, ["p1", "p2"])



###############################################################################

# Ideas for more tests:

# Check that parameter estimates are inside the CIs, and check the flags for this

# Test normalising to different lines repeatedly, and checking that the
# unnecessary interpolated grids are deleted.

# Check coverage of the code, to see what isn't being run?







if __name__ == "__main__":
    print("\nTesting NebulaBayes version {0} ...\n".format(__version__))
    unittest.main(verbosity=2)

