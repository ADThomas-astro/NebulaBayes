"""
    BIGELM: Bayesian Inference with photoionisation model Grids and Emission Line
            Measurements
            (Compulsory contrived acronym, as is the norm in modern astronomy)

    This function performs Bayesian parameter estimation.  The data are a set
    of emission line flux measurements with associated errors.  The model
    is a photoionisation model, varied in a grid over n=2 or more parameters,
    input as n-dimensional grids of fluxes for each emission line.  The measured
    and modelled emission line fluxes are compared to calculate a "likelihood"
    probability distribution, before Bayes' Theroem is applied to produce an
    n-dimensional "posterior" probability distribution for the values of the
    parameters.

    Bigelm is heavily based on IZI (Blanc+2015).


    """


from .src.B0_Main import Bigelm_model
from ._version import __version__

# N.B. The docstring at the top may be accessed interactively in ipython3 with:
# >>> import bigelm
# >>> bigelm?


