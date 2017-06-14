from __future__ import print_function, division
from collections import OrderedDict as OD
import itertools
import os
import sys
import unittest
import numpy as np
import pandas as pd

# Some work to ensure we can import NebulaBayes:
this_file_dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.split(os.path.split(this_file_dir_path)[0])[0])
from NebulaBayes import NB_Model, __version__



"""
Unit test suite to test NebulaBayes
Adam D. Thomas 2017
"""

clean_up = True  # Delete test outputs after running?
test_dir = os.path.join(this_file_dir_path, "test_outputs") # For outputs



# Note about output structure:
# print("NB_Model dir:", [i for i in dir(self.NB_Model_1) if i[0] != "_"])
# ['Interpd_grids', 'Raw_grids']
# print("NB_Result dir:", [i for i in dir(self.Result) if i[0] != "_"])
# Result dir: ['DF_obs', 'Grid_spec', 'Likelihood', 'Plotter', 'Posterior',
#              'Prior', 'deredden', 'obs_flux_arrs', 'obs_flux_err_arrs']




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
        """ Remove the grid file when tests in this class have finished """
        if clean_up:
            os.remove(cls.grid_file)
            if hasattr(cls, "posterior_plot"):
                os.remove(cls.posterior_plot)



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
        kwargs = {"posterior_plot":cls.posterior_plot, "norm_line":"L1"}

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
        # print("Result dir:", [i for i in dir(self.Result) if i[0] != "_"])
        # print("Posterior dir:", [i for i in dir(self.Result.Posterior) if i[0] != "_"])
        # print("DF_estimates", DF_est)


    def test_raw_Grid_spec(self):
        """ Ensure the raw grid spec is as expected """
        # print("Raw_grids dir:", [i for i in dir(self.NB_Model_1.Raw_grids) if i[0] != "_"])
        # ['grid_rel_error', 'grids', 'n_gridpoints', 'ndim', 'paramName2ind',
        # 'paramName2paramMinMax', 'paramName2paramValueArr', 'paramNameAndValue2arrayInd',
        # 'param_names', 'param_values_arrs', 'shape']
        RGrid_spec = self.NB_Model_1.Raw_grids
        self.assertEqual(RGrid_spec.param_names, self.params)
        self.assertEqual(RGrid_spec.ndim, len(self.params))
        self.assertEqual(RGrid_spec.shape, self.n_gridpts_list)
        self.assertEqual(RGrid_spec.n_gridpoints, np.product(self.n_gridpts_list))
        for a1, a2 in zip(RGrid_spec.param_values_arrs, self.val_arrs.values()):
            self.assertTrue(np.allclose(np.asarray(a1), np.asarray(a2)))


    def test_interpolated_Grid_spec(self):
        """ Ensure the interpolated grid spec is as expected """
        # print("Gridspec dir:", [i for i in dir(self.Result.Grid_spec) if i[0] != "_"])
        # ['double_indices', 'double_names', 'n_gridpoints', 'ndim',
        # 'paramName2ind', 'paramName2paramMinMax', 'paramName2paramValueArr',
        # 'paramNameAndValue2arrayInd', 'param_display_names', 'param_names',
        # 'param_values_arrs', 'shape']
        IGrid_spec = self.Result.Grid_spec
        self.assertEqual(IGrid_spec.param_names, self.params)
        self.assertEqual(IGrid_spec.param_display_names, self.params)
        self.assertEqual(IGrid_spec.shape, tuple(self.interpd_shape))
        self.assertEqual(IGrid_spec.n_gridpoints, np.product(self.interpd_shape))


    # def test_grid_interpolation(self):
    #     """ Check some aspects of the grid interpolation """
    #     pass




#  THIS ISN'T VERY USEFUL YET - NEDD A WAY TO CHECK POSTERIOR!
class Test_Obs_from_nonPeak_Gridpoint_2D_Grid_2_Lines(Base_2D_Grid_2_Lines):
    """
    Test for a grid from Base_2D_Grid_2_Lines:  Take a gridpoint that is NOT at
    the peak of the Gaussian ball of emission line fluxes, and check that
    treating these fluxes as observations leads to correct estimates from
    NebulBayes.
    """
    test_gridpoint = [6, 4]  # From zero.  [11, 9] total gridpoints in each dim,
                             # the line peak is at line_peaks = [8, 5]
    
    @classmethod
    def setUpClass(cls):
        super(Test_Obs_from_nonPeak_Gridpoint_2D_Grid_2_Lines, cls).setUpClass()
        test_pt = OD(zip(cls.params, cls.test_gridpoint)) # Map params to gridpt indices
        obs_fluxes = extract_grid_fluxes_i(cls.DF, test_pt, ["L1", "L2"])
        obs_errors = [f / 7. for f in obs_fluxes]

        cls.posterior_plot = os.path.join(test_dir, cls.__name__ + "_posterior.pdf")
        kwargs = {"posterior_plot":cls.posterior_plot, "norm_line":"L1"}

        cls.Result = cls.NB_Model_1(obs_fluxes, obs_errors, cls.lines, **kwargs)

    def test_output_deredden_flag(self):
        self.assertTrue(self.Result.deredden is False)

    def test_parameter_estimates(self):
        """ Ensure the parameter estimates are as expected """
        DF_est = self.Result.Posterior.DF_estimates
        self.assertTrue(all(p in DF_est.index for p in self.params))
        # THE POSTERIOR IS SHAPED LIKE A DONUT.  CHECK FOR A SINGLE LOCAL MIN?





# Ideas for more tests:

# Test in more dimensions, i.e. 3 or 4

# - multiple lines that are simply linear, but with different gradients and
# so on...

# - Check calculation of line ratio priors

# Check that parameter estimates are inside the CIs, and check the flags for this

# Check coverage of the code, to see what isn't being run?







if __name__ == "__main__":
    print("\nTesting NebulaBayes version {0} ...\n".format(__version__))
    unittest.main()
