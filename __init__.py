"""
    BIGELM: Bayesian comparison of photoIonisation model Grids to Emission Line
            Measurements
            (Compulsory contrived acronym, as is the norm in modern astronomy)

    This function performs Bayesian parameter estimation.  The data are a set
    of emission line flux measurements with associated errors.  The model
    is a photoionisation model, varied over n=2 or more parameters, input as
    an n-dimensional grid of predicted emission line fluxes.  The measured
    and modelled emission line fluxes are compared to calculate a "likelihood"
    probability distribution, before Bayes' Theroem is applied to produce an
    n-dimensional "posterior" probability distribution for the values of the
    parameters.

    Bigelm is heavily based on IZI (Blanc+2015).

    Returns a Bigelm_container object (defined in this module), which contains
    the following attributes...

    Optionally the returned class instance may also have attributes containing
    the model grid information (see output_grids keyword below).

    Required arguments to bigelm:
    obs_fluxes:         a numpy array of observed emission-line fluxes
                        normalised to Hbeta
    obs_flux_errors:    a numpy array of corresponding measurement errors
    obs_emission_lines: a list of corresponding emission line names as strings
    and EITHER the following two keyword arguments:
        grid_file:      the filename of an ASCII csv table containing photoionisation model
                        grid data in the form of a database table.
                        Each gridpoint (point in parameter space) is a row in this table.
                        The values of the grid parameters for each row are defined in a column
                        for each parameter.
                        No assumptions are made about the order of the gridpoints (rows) in the table.
                        Spacing of grid values along an axis may be uneven, 
                        but the full grid is required to a be a regular, n-dimensional rectangular grid.
                        There is a column of fluxes for each modelled emission line, and model fluxes
                        are assumed to be normalised to Hbeta
                        Any non-finite fluxes (e.g. nans) will be set to zero.
        grid_params:    List of the unique names of the grid parameters as strings.
                        The order is the order of the grid dimensions, i.e. the order
                        in which arrays in bigelm will be indexed.
    OR the following single keyword argument:
        Grid_container: An instance of the Bigelm_container class defined in
                        this module, containing grid data.  This will either be
                        the output of a previous run of bigelm (which had
                        output_grids=True; see below), or a pre-prepared output
                        of the initialise_grids function in this module.
                        Using this keyword will ensure bigelm uses the
                        previously calculated "Params", "Raw_grids" and
                        "Interpd_grids" attributes of the Bigelm_container class
                        instance.  Avoiding recalculation of the raw and interpolated
                        flux grids for each emission line saves a very large fraction of the bigelm
                        computation time.  Note that the "Params" attribute is an instance of the
                        Grid_parameters class defined in this module; the "Raw_grids" and "Interpd_grids"
                        attributes are instances of the Bigelm_grid class defined in this module.  Any 
                        contents of Grid_container other than these three attributes are ignored.

    Optional additional keyword arguments:
    image_out:            A filename for saving out a results image of 2D and 1D marginalised posterior pdfs.
                          The figure will only be generated and saved if this keyword parameter is specified.
    output_grids:         A Boolean, False by default.  If True, the outputted Results object will
                          contain the raw grids ("Raw_grids") and interpolated grids ("Interpd_grids") objects
                          as attributes.  Note that the interpolated grids object may be large (e.g. 6 Mb * 50 lines)
                          The Raw_grids and Interpd_grids objects are instances of the Bigelm_grid class defined in this module.
    interpd_grid_shape:   A tuple of integers, giving the size of each dimension of the interpolated
                          grid.  The order of the integers corresponds to the order of parameters in grid_params.
                          The default is 30 gridpoints along each dimension.  Note that the number of
                          interpolated gridpoints entered in interpd_grid_shape
                          may have a major impact on the speed of the program.
                          This keyword may only be supplied if the "Grid_container" keyword is not used.
                          Will be passed to function initialise_grids.
    param_display_names:  A dictionary of display names for grid parameters, for plotting purposes.
                          A dictionary key is the parameter name in the grid file, and the corresponding
                          value its display name.
                          Can be raw strings (i.e. r"string") in order to include e.g. Greek letters.
                          Not all of the grid parameters need to be included in param_display_names;
                          raw parameter names will be used as display names by default.
                          This keyword may only be supplied if the "Grid_container" keyword is not used
                          (the previous param_display_names will be used).
    #priors???            NOT IMPLETMENTED: A dictionary of functions for priors for each parameter... ???
    #extra_lines:         NOT IMPLEMENTED: A list of additional emission lines to make grids for.  Useful for
                          making grids for use in later bigelm runs.  A union of obs_emission_lines and extra_lines
                          is used to make the final list of emission lines for grids.


    Other notes:
    We don't deal with measured emission-line fluxes that are provided summed for two lines.
    We ignore the possibility of including only upper bounds for emission-line fluxes
    We currently assume uniform priors for everything
    We currently compare linear fluxes, not log fluxes...
    In calculating the likelihood we assume a systematic error on the normalised model fluxes of 0.15dex.
    In finding marginalised posteriors, we use trapezium integration - perhaps dodgy, 
    but I think we have too few dimensions for Monte Carlo methods and I don't think it's
    worth doing higher-order numerical integration.
    At the moment zero model flux => zero systematic error on the flux. Wrong!

    """


from .src.bigelm_main import bigelm
from ._version import __version__

# N.B. The docstring at the top may be accessed interactively in ipython3 with:
# >>> import bigelm
# >>> bigelm?


