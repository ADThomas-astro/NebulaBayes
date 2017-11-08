from __future__ import print_function, division
import os
# import numpy as np
# import pandas as pd
from NebulaBayes import NB_Model


"""
This script shows examples of basic usage of NebulaBayes.  The code works in
python 2 and python 3.

There are examples of how to:
 - Measure the metallicity, ionisation parameter and pressure from HII-region
   fluxes that have already been dereddened
 - Measure the metallicity, ionisation parameter, pressure and hardness of the
   ionising radiation from AGN NLR fluxes that have already been dereddened

This script may be run unchanged to save output in the NebulaBayes/docs
directory.  Otherwise add a custom "OUT_DIR" below.
"""


# By default save the output files in the NebulaBayes/docs subdirectory,
# assuming this file is still in that directory.
DOCS_PATH = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = DOCS_PATH



##############################################################################
print("\nRunning basic HII example...")
# In this example the NebulaBayes built-in HII region grid is used to constrain
# the three parameter of the grid - oxygen abundance, ionisation parameter
# and pressure.

# These HII-region optical emission-line fluxes have already been dereddened
linelist = ["OII3726_29", "Hbeta", "OIII5007", "OI6300", "Halpha", "NII6583",
            "SII6716", "SII6731"]
obs_fluxes = [8.151, 4.634, 1.999, 0.09562, 13.21, 5.116, 1.377, 1.249]
obs_errs = [0.09008, 0.04013, 0.01888, 0.00222, 0.07635, 0.03159, 0.00999,
            0.00923]
# Fluxes/errors will be normalised to the flux of the default norm_line, Hbeta
# The emission line names match those in the grid (see the NebulaBayes/grids
# directory)

# Initialise the NB_Model, which loads and interpolates the model flux grids:
NB_Model_HII = NB_Model("HII", line_list=linelist)

# Set outputs:
kwargs = {"prior_plot": os.path.join(OUT_DIR, "1_HII_prior_plot.pdf"),
          "likelihood_plot": os.path.join(OUT_DIR, "1_HII_likelihood_plot.pdf"),
          "posterior_plot": os.path.join(OUT_DIR, "1_HII_posterior_plot.pdf"),
          "estimate_table": os.path.join(OUT_DIR, "1_HII_param_estimates.csv"),
          "best_model_table": os.path.join(OUT_DIR, "1_HII_best_model.csv"),
          }

# Run parameter estimation once
Result_HII = NB_Model_HII(obs_fluxes, obs_errs, linelist, **kwargs)
# NB_Model_HII may be called repeatedly to do Bayesian parameter estimation
# on different sets of observed fluxes with the same grid.



##############################################################################
print("\nRunning basic NLR example...")
# In this example we use the NebulaBayes built-in AGN narrow-line region (NLR)
# grid to constrain the four parameter of the grid - oxygen abundance,
# ionisation parameter, pressure, and the hardness of the ionising continuum.

# These NLR optical emission-line fluxes have already been dereddened
linelist = ["OII3726_29", "NeIII3869", "Hgamma", "HeII4686", "Hbeta",
            "OIII5007", "HeI5876", "Halpha", "NII6583", "SII6716", "SII6731"]
obs_fluxes = [4.2162, 1.159, 0.7161, 0.3970, 1.292, 12.88, 0.1597, 3.747,
              5.027, 1.105, 1.198]
obs_errs = [0.5330, 0.2073, 0.1172, 0.0630, 0.1864, 1.759, 0.0225, 0.3919,
            0.5226,  0.1156, 0.1248]

NB_Model_NLR = NB_Model("NLR", line_list=linelist)

kwargs = {"prior_plot": os.path.join(OUT_DIR, "1_NLR_prior_plot.pdf"),
          "posterior_plot": os.path.join(OUT_DIR, "1_NLR_posterior_plot.pdf"),
          "estimate_table": os.path.join(OUT_DIR, "1_NLR_param_estimates.csv"),
          "prior": [("SII6716", "SII6731")],
          }

Result_NLR = NB_Model_NLR(obs_fluxes, obs_errs, linelist, **kwargs)


##############################################################################
print("Basic example script complete.")

