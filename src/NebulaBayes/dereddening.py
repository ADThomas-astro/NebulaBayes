from __future__ import print_function, division
import numpy as np
import unittest

"""
Do reddening and de-reddening of optical emission-line fluxes using the
equations in the appendix of Vogt 2013
http://adsabs.harvard.edu/abs/2013ApJ...768..151V

There are 3 public functions; these are to redden and deredden fluxes using the
Balmer decrement, and to calculate the extinction in magnitudes from the low
and high Balmer decrements.

The method used here is valid for wavelengths in [2480, 12390] Angstroem.

This module may be imported to be used in python code, or run as a script to
run some basic tests.  The code works with both Python 2 and Python 3.

Adam D. Thomas 2017
Research School of Astronomy and Astrophysics
Australian National University
February - March 2017

Keywords - extinction, extragalactic reddening, Fischera & Dopita 2005, 2011,
           attenuation curve, relative color excess, Calzetti 2001, dust
"""
__version__ = 0.7



def _calc_relative_colour_excess(lambda_AA):
    """
    Calculate the "relative colour excess" E_(lambda-V) / E_(B-V) using Equation
    A11 in the appendix of Vogt13.  Valid for lambda in [2480, 12390] Angstroem,
    with the relative color excess decreasing monotonically from 3.86 to -2.79
    over this range in lambda.
    lambda_AA: Wavelength in Angstroems. A list or array of floats.
    Returns the relative colour excess as an array of floats.
    """
    lambda_AA = np.asarray(lambda_AA)
    assert np.all(lambda_AA > 2480) and np.all(lambda_AA < 12390)
    lambda_um = lambda_AA * 1e-4 # Convert wavelength to um

    # Coefficients in Equation A11 in Vogt13
    A11_coeffs = {0:-4.61777, -1:1.41612, -2:1.52077, -3:-0.63269, -4:0.07386}
    # Calculate relative color excess using Equation A11
    return sum( (A11_coeffs[i] * lambda_um**i) for i in [0,-1,-2,-3,-4] )



def _find_BD(line_lambdas, line_fluxes, flux_errs=None):
    """
    Given lists or arrays of emission line wavelengths and corresponding fluxes,
    find the Balmer decrement (BD).  We do some checks.
    line_lambdas: 1D list or array of line wavelengths in Angstroems
    line_fluxes: 1D list or array, or alternatively a list of arrays of arbitrary
                 shape (but all the same shape), containing line fluxes.
    flux_errs: Errors corresponding to line_fluxes; optional.
    Returns the Balmer decrement F_Halpha/F_Hbeta as a float or array of floats.
    If flux_errs is supplied, the error in the Balmer decrement is also returned,
    with the error in Halpha and Hbeta propagated into the Balmer decrement.
    """
    line_lambdas = np.asarray(line_lambdas)
    assert line_lambdas.size == len(line_fluxes)
    assert line_lambdas.ndim == 1
    if flux_errs is not None:
        assert len(flux_errs) == len(line_fluxes)

    # Find Halpha and Hbeta fluxes:
    rounded_lambdas = np.round(line_lambdas, 0)
    where_Halpha = (rounded_lambdas == 6563) # Halpha is at 6562.819 AA
    where_Hbeta  = (rounded_lambdas == 4861) # Hbeta is at 4861.333 AA
    n_where_Halpha, n_where_Hbeta = np.sum(where_Halpha), np.sum(where_Hbeta)
    if n_where_Halpha == 0:
        raise ValueError("Input lines do not include Halpha (6563 AA)")
    elif n_where_Halpha == 1:
        ind_Ha = where_Halpha.nonzero()[0][0] # An integer index
        halpha_flux = line_fluxes[ind_Ha]
        # So halpha_flux is a float if "line_fluxes" is 1D, or an array
        # if "line_fluxes" is a list of arrays.
    else:
        raise ValueError("Multiple lines at a wavelength similar to Halpha!")
    if n_where_Hbeta == 0:
        raise ValueError("Input lines do not include Hbeta (4861 AA)")
    elif n_where_Hbeta == 1:
        ind_Hb = where_Hbeta.nonzero()[0][0] # An integer index
        hbeta_flux = line_fluxes[ind_Hb]
    else:
        raise ValueError("Multiple lines at a wavelength similar to Hbeta!")

    decrement = halpha_flux / hbeta_flux
    if flux_errs is None:
        # Return Balmer decrement F_Halpha / F_Hbeta (possibly and array of
        # Balmer decrements)
        return decrement
    else: # Also return the error in the Balmer decrement.
        # We propagate the errors in Halpha and Hbeta
        halpha_rel_err = flux_errs[ind_Ha] / halpha_flux
        hbeta_rel_err  = flux_errs[ind_Hb] / hbeta_flux
        decrement_err = decrement * np.hypot(halpha_rel_err, hbeta_rel_err)
        return decrement, decrement_err



def _apply_BD(line_lambdas, line_fluxes, flux_errs, BD, normalise):
    """
    De-redden or redden emission line fluxes by considering a target Balmer
    decrement (BD), using Equation A12 in the Appendix of Vogt13.  This function
    is its own inverse, i.e. reddens or dereddens.
    line_lambdas, line_fluxes, flux_errs: As in the functions "deredden" and "redden".
    BD: Desired Balmer decrement (F_Halpha / F_Hbeta) to use in (de)reddening
        (will be the Balmer decrement in the output).  If BD is a scalar, the
        output will be a 1D numpy array.  If BD is an array, the (de)reddening
        will be applied for all supplied values of BD and the output will be a
        list of flux arrays.  The output arrays will all have the same shape as
        BD and the list will have the same length as line_lambdas.
    normalise: Boolean.  Normalise output to Hbeta==1?

    Returns out_fluxes, an array or list of arrays of (de)reddened line fluxes
    corresponding to the input line_lambdas.  If flux_errs are supplied then the
    output will be (out_fluxes, out_flux_errs), where out_flux_errs has the same
    format as out_fluxes.
    """
    if BD is None:
        raise ValueError("BD must be specified")
    line_lambdas, in_fluxes = np.asarray(line_lambdas), np.asarray(line_fluxes)
    assert line_lambdas.ndim == 1
    assert line_lambdas.shape == in_fluxes.shape
    if flux_errs is not None:
        in_flux_errs = np.asarray(flux_errs)
    BD2 = np.asarray(BD)
    is_multiple = (BD2.size > 1) # Output multiple fluxes for each line?
    # Find observed Balmer decrement:
    if flux_errs is None:
        BD1 = _find_BD(line_lambdas, in_fluxes)
    else:
        BD1, BD1_err = _find_BD(line_lambdas, in_fluxes, in_flux_errs)

    # Apply Equation A12 in Vogt13
    r_c_e = _calc_relative_colour_excess(line_lambdas) # Vector of RCE
    p = 0.76*(r_c_e + 4.5) # Vector of exponents (p=Powers) for (de)reddening equation
    # Fluxes:
    if is_multiple: # BD2 is an array; construct a list of (de)reddened flux arrays
        out_fluxes = [f * (BD1 / BD2)**p_i for f,p_i in zip(in_fluxes, p)]
    else: # BD2 is a scalar; make a 1D vector of (de)reddened fluxes
        out_fluxes = in_fluxes * (BD1 / BD2)**p
    # Errors: propagate errors on the fluxes and on the starting Balmer decrement
    #     f2 = f1 * (BD1/BD2)^p  where f1 and B1 have uncertainties.
    # => df2 = f1 BD2**-p |p BD1**(p-1)| dBD1  +  (BD1/BD2)**p df1
    if flux_errs is not None and is_multiple:
        out_errs = [(   f * BD2**-p_i * np.abs(p_i * BD1**(p_i-1)) * BD1_err
                      + (BD1 / BD2)**p_i * f_err ) for f, f_err, p_i in zip(
                                                    in_fluxes, in_flux_errs, p)]
    elif flux_errs is not None:
        out_errs = ( in_fluxes * BD2**-p * np.abs(p * BD1**(p-1)) * BD1_err
                      + (BD1 / BD2)**p * in_flux_errs                       )

    if normalise:
        where_Hbeta = (np.round(line_lambdas, 0) == 4861) # Hbeta is at 4861.333 AA
        # We've already checked, and there's exactly one line qualifying as Hbeta
        ind_Hb = where_Hbeta.nonzero()[0][0] # An integer index
        hbeta_unnormed_out_flux = out_fluxes[ind_Hb].copy()
        # The "copy" on the above line is important!
        # We don't propagate errors in the normalisation
        # Scale fluxes:
        if is_multiple:
            for f in out_fluxes:
                f /= hbeta_unnormed_out_flux # In-place division
        else:
            out_fluxes /= hbeta_unnormed_out_flux
        # Scale errors:
        if flux_errs is not None:
            if is_multiple:
                for err in out_errs:
                    err /= hbeta_unnormed_out_flux # In-place division
            else:
                out_errs /= hbeta_unnormed_out_flux

    if flux_errs is None:
        return out_fluxes
    else:
        return out_fluxes, out_errs



def _BD_from_Av_for_dereddening(line_lambdas, line_fluxes, A_v):
    """
    Find the de-reddened Balmer decrement (BD) that would arise from "removing"
    an extinction of A_v (magnitudes) from the line_fluxes.
    line_lambdas, line_fluxes: As in the function "deredden".
    A_v: The extinction (magnitudes), as a scalar or array of extinction values.

    Returns the Balmer decrement dereddened_BD (F_Halpha / F_Hbeta), as a float
    or array of floats with the same shape as A_v.
    """
    assert np.all(np.asarray(A_v) >= 0)
    initial_BD = _find_BD(line_lambdas, line_fluxes)

    # Calculate the Balmer decrement (BD) that would result from "removing" an
    # extinction of A_v, using an inverted form of Equation A14 in Vogt13.
    dereddened_BD = initial_BD / 10**(A_v / 8.55)
    return dereddened_BD



def _BD_from_Av_for_reddening(line_lambdas, line_fluxes, A_v):
    """
    Find the reddened Balmer decrement (BD) that would arise from "applying"
    an extinction of A_v (magnitudes) to the line_fluxes.
    line_lambdas, line_fluxes: As in the function "redden".
    A_v: The extinction (magnitudes), as a scalar or array of extinction values.

    Returns the Balmer decrement reddened_BD (F_Halpha / F_Hbeta), as a float
    or array of floats with the same shape as A_v.
    """
    assert np.all(np.asarray(A_v) >= 0)
    initial_BD = _find_BD(line_lambdas, line_fluxes)

    # Calculate the Balmer decrement (BD) that would result from an extinction
    # of A_v, using an inverted form of Equation A14 in Vogt13.
    reddened_BD = initial_BD * 10**(A_v / 8.55)
    # This equation differs slightly from that in "_BD_from_Av_for_dereddening"!
    return reddened_BD



def Av_from_BD(BD_low, BD_high):
    """
    Calculate the extinction in magnitudes associated with in increase in the 
    Balmer decrement, BD = F_Halpha / F_Hbeta.
    BD_low: The intrinsic (lower) Balmer decrement
    BD_high: The reddened (higher) Balmer decrement
    If both BD_low and BD_high are non-scalar arrays, they must have the same shape.

    Returns A_v, the extinction in magnitudes as a float or array of floats.
    """
    BD_low_arr, BD_high_arr = np.asarray(BD_low), np.asarray(BD_high)
    if BD_low_arr.size > 1 and BD_high_arr.size > 1:
        assert BD_low_arr.shape == BD_high_arr.shape
    assert np.all(BD_low_arr <= BD_high_arr)
    
    A_v = 8.55 * np.log10( BD_high / BD_low ) # Equation A14 in Vogt13
    return A_v



def deredden(line_lambdas, line_fluxes, line_errs=None, BD=None, A_v=None,
                                                               normalise=False):
    """
    Deredden emission line fluxes by either specifying a target Balmer decrement
    (F_Halpha/F_Hbeta), or an extinction A_v to be "removed".
    This is the inverse function of "redden".
    line_lambdas: List/array of floats.  Wavelengths in Angstroems.
    line_fluxes:  Corresponding float, list or array of reddened fluxes. Note
                  that line_lambdas and line_fluxes must include Halpha
                  (6562.8 A) and Hbeta (4861.3 A).
    line_errs:    Flux errors corresponding to line_fluxes (optional).
    A_v: The assumed extinction (magnitudes) to "remove" when dereddening.
    BD:  The target (intrinsic) Balmer decrement.
    Only one of "A_v" or "BD" may be specified.  If neither is set, BD=2.85 is
    used.  If the supplied value is a scalar, the output fluxes will be a 1D
    numpy array.  If BD (or A_v) is an array, the de-reddening will be applied
    for all supplied values of BD (or A_v) and the output fluxes will be a list
    of arrays.  The output arrays will all have the same shape as BD (or A_v)
    and the list containing them will have the same length as line_lambdas.
    normalise: Normalise output to Hbeta==1?  Default:False

    Returns dered_fluxes if line_errs is not specified, or alternatively returns
    (dered_fluxes, dered_errs) if line_errs is given.  The array (or list of
    arrays) dered_fluxes contains de-reddened fluxes corresponding to the input
    line_lambdas.  The output dered_errs contains corresponding errors propagated
    from the input errors.  The dereddened fluxes (and errors) are normalised to
    Hbeta == 1 if normalise == True. 
    """
    # Look at what was specified, and determine the Balmer decrement to use.
    if A_v is None:
        if BD is None:
            BD = 2.85
    else: # A_v is specified
        if BD is not None:
            raise ValueError("Must specify only one of A_v or BD, not both")
        BD = _BD_from_Av_for_dereddening(line_lambdas, line_fluxes, A_v)

    return _apply_BD(line_lambdas, line_fluxes, line_errs, BD=BD, normalise=normalise)



def redden(line_lambdas, line_fluxes, line_errs=None, BD=None, A_v=None,
                                                               normalise=False):
    """
    Redden emission line fluxes by either specifying a target Balmer decrement
    (F_Halpha/F_Hbeta), or an extinction A_v to be "applied".
    This is the inverse function of "deredden".
    line_lambdas: Float, or list/array of floats.  Wavelengths in Angstroems.
    line_fluxes:  Corresponding float, list or array of intrinsic fluxes. Note
                  that line_lambdas and line_fluxes must include Halpha
                  (6562.8 A) and Hbeta (4861.3 A).
    line_errs:    Flux errors corresponding to line_fluxes (optional).
    A_v: The assumed extinction (magnitudes) to "apply" when reddening.
    BD:  The target Balmer decrement.
    Only one of "A_v" or "BD" may be specified.  If the supplied value is a
    scalar, the output fluxes will be a 1D numpy array.  If BD (or A_v) is an
    array, the reddening will be applied for all supplied values of BD (or A_v)
    and the output fluxes will be a list of arrays.  The output arrays will all
    have the same shape as BD (or A_v) and the list containing them will have
    the same length as line_lambdas.
    normalise: Normalise output to Hbeta==1?  Default:False

    Returns red_fluxes if line_errs is not specified, or alternatively returns
    (red_fluxes, red_errs) if line_errs is given.  The array (or list of
    arrays) red_fluxes contains reddened fluxes corresponding to the input
    line_lambdas.  The output red_errs contains corresponding errors propagated
    from the input errors.  The reddened fluxes (and errors) are normalised to
    Hbeta == 1 if normalise == True.
    """
    # Look at what was specified, and determine the balmer decrement to use.
    if A_v is None:
        if BD is None:
            raise ValueError("Must specify one of A_v or BD")
    else: # A_v is specified
        if BD is not None:
            raise ValueError("Must specify only one of A_v or BD, not both")
        BD = _BD_from_Av_for_reddening(line_lambdas, line_fluxes, A_v)

    return _apply_BD(line_lambdas, line_fluxes, line_errs, BD=BD, normalise=normalise)



class _Tests(unittest.TestCase):
    """ A collection of test cases for testing this module """

    def test_simple_red_dered_single_BD(self):
        # Test that the Balmer decrement behaves as expected
        l_HbHa = [4861, 6563] # Hbeta and Halpha wavelengths in Angstroems
        self.assertTrue(np.isclose(
            _apply_BD(l_HbHa,[1.0,3.5], None, BD=2.9, normalise=True)[1], 2.9, atol=atol))
        self.assertTrue(np.isclose(
            _apply_BD(l_HbHa,[1.0,2.9], None, BD=3.5, normalise=True)[1], 3.5, atol=atol))
        self.assertTrue(np.isclose(
            deredden( l_HbHa,[1.0,3.5], None, BD=2.9, normalise=True)[1], 2.9, atol=atol))
        self.assertTrue(np.isclose(
            redden(   l_HbHa,[1.0,2.9], None, BD=3.5, normalise=True)[1], 3.5, atol=atol))


    def test_simple_red_dered_multiple_BD(self):
        # Test that the Balmer decrement behaves as expected
        l_HbHa = [4861, 6563] # Hbeta and Halpha wavelengths in Angstroems
        arr_29, arr_31 = np.array([2.9, 2.9, 2.9]), np.array([3.1, 3.1, 3.1])
        self.assertTrue(np.allclose(
            _apply_BD(l_HbHa,[1.0,3.5], None, BD=arr_29, normalise=True)[1], arr_29, atol=atol))
        self.assertTrue(np.allclose(
            _apply_BD(l_HbHa,[1.0,2.9], None, BD=arr_31, normalise=True)[1], arr_31, atol=atol))
        self.assertTrue(np.allclose(
            deredden( l_HbHa,[1.0,3.5], None, BD=arr_29, normalise=True)[1], arr_29, atol=atol))
        self.assertTrue(np.allclose(
            redden(   l_HbHa,[1.0,2.9], None, BD=arr_31, normalise=True)[1], arr_31, atol=atol))


    def test_1D_outputs(self):
        # Some simple tests of the functionality of this module for 1D outputs
        # (i.e. reddening or dereddening by only one input BD or A_v)

        # Test data
        BD1, BD_intrinsic = 3.41, 2.9
        obs_lambdas = [6563, 6583, 3726.032, 4861.33] # Halpha, [NII], [OII], Hbeta
        obs_fluxes  = [BD1,  4.1,  1.35,     0.99999]

        # Test that "_apply_BD" is its own inverse function
        dered_fluxes_1 = _apply_BD(obs_lambdas, obs_fluxes, None, BD=BD_intrinsic, normalise=True)
        rered_fluxes_1 = _apply_BD(obs_lambdas, dered_fluxes_1, None, BD=BD1, normalise=True)
        self.assertTrue(np.allclose(rered_fluxes_1, obs_fluxes, atol=atol))
        dered_fluxes_1a = _apply_BD(obs_lambdas, obs_fluxes, None, BD=BD_intrinsic, normalise=False)
        rered_fluxes_1a = _apply_BD(obs_lambdas, dered_fluxes_1a, None, BD=BD1, normalise=True)
        self.assertTrue(np.allclose(rered_fluxes_1a, obs_fluxes, atol=atol))

        # Test that "_BD_from_Av_for_dereddening", "_BD_from_Av_for_dereddening"
        # and "Av_from_BD" are consistent
        Av1 = Av_from_BD(BD_low=BD_intrinsic, BD_high=BD1)
        BD_a = _BD_from_Av_for_dereddening(obs_lambdas, obs_fluxes, A_v=Av1)
        self.assertTrue(np.isclose(BD_a, BD_intrinsic, atol=atol))
        BD_b = _BD_from_Av_for_reddening(obs_lambdas, dered_fluxes_1, A_v=Av1)
        self.assertTrue(np.isclose(BD_b, BD1, atol=atol))

        # Test that "deredden" and "redden" are inverse functions, using A_v
        dered_fluxes_2 = deredden(obs_lambdas, obs_fluxes, A_v=Av1, normalise=True)
        self.assertTrue(np.allclose(dered_fluxes_2, dered_fluxes_1, atol=atol))
        rered_fluxes_2 = redden(obs_lambdas, dered_fluxes_2, A_v=Av1, normalise=True)
        self.assertTrue(np.allclose(rered_fluxes_2, rered_fluxes_1, atol=atol))
        self.assertTrue(np.allclose(rered_fluxes_2, obs_fluxes, atol=atol))

        # Test that an extinction of 0 results in negligible flux change
        same_fluxes = redden(obs_lambdas, obs_fluxes, A_v=0, normalise=True)
        self.assertTrue(np.allclose(same_fluxes, obs_fluxes, atol=atol))


    def test_nD_outputs(self):
        # Some simple tests of the functionality of this module for nD outputs
        # (i.e. reddening or dereddening for all of an array of BD or A_v)

        # Test data
        BD1, BD_intrinsic = 3.41, np.array([[2.85, 2.9, 2.95],[3, 3.05, 3.1]])
        obs_lambdas = [6563, 6583, 3726.032, 4861.33] # Halpha, [NII], [OII], Hbeta
        obs_fluxes  = [BD1,  4.1,  1.35,     0.99999]

        # Test that "_apply_BD" and "_apply_BD" are inverse functions
        dered_fluxes_1 = _apply_BD(obs_lambdas, obs_fluxes, None, BD=BD_intrinsic, normalise=True)
        # As a list of 1D tuples of dereddened [Halpha, [NII], [OII], Hbeta] fluxes:
        dered_fluxes_1a = list(zip(*[a.ravel() for a in dered_fluxes_1]))
        rered_fluxes_1a = [_apply_BD(obs_lambdas, f, None, BD=BD1,
                                              normalise=True) for f in dered_fluxes_1a]
        for rered_fluxes_1_i in rered_fluxes_1a:
            self.assertTrue(np.allclose(rered_fluxes_1_i, obs_fluxes, atol=atol))

        # Test that "_BD_from_Av_for_dereddening", "_BD_from_Av_for_dereddening"
        # and "Av_from_BD" are consistent
        Av1 = Av_from_BD(BD_low=BD_intrinsic, BD_high=BD1) # Array of A_v
        BD_a = _BD_from_Av_for_dereddening(obs_lambdas, obs_fluxes, A_v=Av1)
        self.assertTrue(np.allclose(BD_a, BD_intrinsic, atol=atol))
        for i, dered_fluxes_1_i in enumerate(dered_fluxes_1a):   
            BD_b = _BD_from_Av_for_reddening(obs_lambdas, dered_fluxes_1_i, A_v=Av1.flat[i])
            self.assertTrue(np.allclose(BD_b, BD1, atol=atol))

        # Test that "deredden" and "redden" are inverse functions, using A_v
        dered_fluxes_2 = deredden(obs_lambdas, obs_fluxes, A_v=Av1, normalise=True)
        dered_fluxes_2a = list(zip(*[a.ravel() for a in dered_fluxes_2]))
        for i, dered_fluxes_1_i in enumerate(dered_fluxes_1a):
            self.assertTrue(np.allclose(dered_fluxes_2a[i], dered_fluxes_1_i, atol=atol))
            rered_fluxes_2_i = redden(obs_lambdas, dered_fluxes_2a[i],
                                                    A_v=Av1.flat[i], normalise=True)
            self.assertTrue(np.allclose(rered_fluxes_2_i, rered_fluxes_1a[i], atol=atol))
            self.assertTrue(np.allclose(rered_fluxes_2_i, obs_fluxes, atol=atol))


    def test_uncertainty_handling_1D(self):
        # Test uncertainty handling in this module, for cases with 1D outputs

        # We can rewrite the formula for the uncertainty (in _apply_BD) as follows:
        # df2 = f2 / SN1 * (p+1) where SN1 = f_Ha / dF_Ha
        # where we're only dereddening Halpha, and assuming the uncertainty in
        # the initial BD is the uncertainty in the initial Ha flux.
        # The maximum value of p+1 is 7.36 (for rce=3.86 for lambda=2480AA), so an
        # upper bound is df_Ha2 < 7.36 * f2 / (f1 / err1), where df_Ha2 is not
        # normalised to Hbeta == 1 (which would decrease df_Ha2).

        for BD2 in [2.9, 5.3]:  # De-reddening and reddening
            l1, f1, err1 = [4861, 6563], [1.0, 3.5], [0, 0.81]  # Hbeta, Halpha
            # Set zero error in Hbeta for convenience
            f2_l, err2_l = _apply_BD(l1, f1, err1, BD=BD2, normalise=True)
            f2, err2 = f2_l[1], err2_l[1]
            # Check that errors are in expected range
            self.assertTrue(err2 > (f2 / f1[1]) * err1[1]) # Check lower bound (no error propagation)
            SN1 = f1[1] / err1[1] # Signal-to-noise
            self.assertTrue(err2 < 7.36 * f2 / SN1) # Check upper bound on error
            p_Ha = 0.76 * (_calc_relative_colour_excess(l1[1]) + 4.5) # Exponent
            # Check actual value using alternative equation:
            # Note that f2 has already been scaled by Hbeta, so now the equation 
            # df2 = f2 / SN1 * (p+1) will give the error normalised to Hbeta == 1.
            self.assertTrue(np.isclose(err2, (f2 / SN1 * (p_Ha+1)), atol=atol))


    def test_uncertainty_handling_nD(self):
        # Test uncertainty handling in this module, for cases with nD outputs
        BDa = np.array([[2.9, 3.0], [2.8, 2.7], [2.6,2.85]])
        for BD2 in [BDa, BDa + 2.5]:  # De-reddening and reddening
            l1, f1, err1 = [4861, 6563], [1.0, 3.5], [0, 0.81]  # Hbeta, Halpha
            # Make lists of arrays:
            f2, err2 = _apply_BD(l1, f1, err1, BD=BD2, normalise=True)
            # Check that errors are in expected range
            # Check lower bound (no error propagation):
            self.assertTrue(np.all(err2[1] > (f2[1] / f1[1]) * err1[1]))
            SN1 = f1[1] / err1[1] # Signal-to-noise
            self.assertTrue(np.all(err2[1] < 7.36 * f2[1] / SN1)) # Check upper bound on error
            p_Ha = 0.76 * (_calc_relative_colour_excess(l1[1]) + 4.5) # Exponent
            self.assertTrue(np.allclose(err2[1], (f2[1] / SN1 * (p_Ha+1)), atol=atol))



if __name__ == "__main__":
    # If we run this module as a script, do some tests
    print("Testing ADT dereddening module version {0} ...".format(__version__))
    atol = 5e-4  # Absolute tolerance for comparing fluxes (Hbeta == 1)
    # Decrease this tolerance to watch the unit tests fail!
    unittest.main()

