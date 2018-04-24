"""
NebulaBayes is a package for astronomers that aims to provide a very general
way to compare observed emission line fluxes to model predictions, in order to
constrain physical parameters such as the nebular metallicity.

NebulaBayes is provided with two photoionization model grids produced using the
MAPPINGS 5.1 model.  One grid is a 3D HII-region grid which may be used to
constrain the oxygen abundance (12 + log O/H), ionisation parameter (log U) and
gas pressure (log P/k).  The other grid is for AGN narrow-line regions (NLRs)
and has 4 dimensions, with the added parameter "log E_peak" being a measure of
the hardness of the ionising continuum.  NebulaBayes accepts model grids in a
simple table format, and is agnostic to the number of dimensions in the grid,
the parameter names, and the emission line names.

The NebulaBayes.NB_Model class is the entry point for performing Bayesian
parameter estimation.  The class is initialised with a chosen model grid, at
which point the model flux grids are loaded, interpolated, and stored.  The
NB_Model instance may then be called one or more times to run Bayesian
parameter estimation using observed fluxes.  Many outputs are available,
including tables and figures, and all results and working are stored on the
object returned when the NB_Model instance is called.

See the "docs" directory in the installed NebulaBayes package for more
information, suggestions for getting started, and examples. (Type the following
at the terminal to show the location of the installed package):
    $ python -c "import NebulaBayes; print(NebulaBayes.__file__)"

The documentation assumes some knowledge of Bayesian statistics and scientific
python (numpy, matplotlib and pandas).

NebulaBayes is heavily based on IZI (Blanc+ 2015).

If you use NebulaBayes, please cite
http://adsabs.harvard.edu/abs/2018ApJ...856...89T
\bibitem[Thomas et al.(2018)]{2018ApJ...856...89T} Thomas, A.~D., Dopita, M.~A.,
Kewley, L.~J., et al.\ 2018, \apj, 856, 89

Adam D. Thomas
Research School of Astronomy and Astrophysics
Australian National University
adam.thomas AT anu.edu.au
2015 - 2018
"""


from .NB0_Main import NB_Model, NB_logger
from ._version import __version__

# N.B. The docstring at the top may be accessed interactively in ipython3 with:
# >>> import NebulaBayes
# >>> NebulaBayes?