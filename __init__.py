"""
    NebulaBayes
    Adam D. Thomas
    Research School of Astronomy and Astrophysics
    Australian National University
    2015 - 2017

    This module performs Bayesian parameter estimation.  The data are a set
    of emission line flux measurements with associated errors.  The model
    is a photoionisation model, varied in a grid over n=2 or more parameters,
    input as n-dimensional grids of fluxes for each emission line.  The model is
    for an HII region or AGN Narrow Line Region, for example.  The measured
    and modelled emission line fluxes are compared to calculate a "likelihood"
    probability distribution, before Bayes' Theroem is applied to produce an
    n-dimensional "posterior" probability distribution for the values of the
    parameters.  The parameter values are estimated from 1D marginalised
    posteriors.

    Bigelm is heavily based on IZI (Blanc+2015).
    """


from .src.NB0_Main import Bigelm_model
from ._version import __version__

# N.B. The docstring at the top may be accessed interactively in ipython3 with:
# >>> import bigelm
# >>> bigelm?


