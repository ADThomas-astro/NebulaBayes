from __future__ import print_function, division
import numpy as np

"""
Do reddening and de-reddening of optical emission-line fluxes using the
equations in the appendix of Vogt 2013
http://adsabs.harvard.edu/abs/2013ApJ...768..151V

There are 3 public functions; these are to redden and deredden fluxes using the
Balmer decrement, and to calculate the extinction in magnitudes from the low
and high Balmer decrements.

This module may be imported to be used in python code, or run as a script to
run some basic tests.  The code works with both Python 2 and Python 3.

Adam D. Thomas 2017
Research School of Astronomy and Astrophysics
Australian National University

History
 - 20170214 - Original version
 - 20170215 - Vectorised all functions; added "deredden_Av" function; gave both
              public "BD" functions the same API (barring argument names);
              tweaked tests; did a manual test of reddening against RSS's
              "dered-106" command-line tool.  Results seemed reasonable.
 - 20170216 - Added "Av_from_BD", "deredden" and "redden" functions; removed
              "(de)redden_Av" functions; refactored tests into the "_test"
              function and expanded them. 
 - 20170221 - Fixed minor bug in checking length of observed errors array


Keywords - extinction, extragalactic reddening, Fischera & Dopita 2005, 2011,
           attenuation curve, relative color excess, Calzetti 2001, dust
"""
__version__ = 0.04



def _calc_relative_colour_excess(lambda_AA):
    """
    Calculate the "relative colour excess" E_(lambda-V) / E_(B-V) using Equation
    A11 in the appendix of Vogt13.
    lambda_AA: Wavelength in Angstroems. A float, list of floats or array of floats.
    Returns the relative colour excess as a float or array of floats.
    """
    lambda_AA = np.array(lambda_AA)
    assert np.all(lambda_AA > 0)
    lambda_um = lambda_AA * 1e-4 # Convert wavelength to um

    # Coefficients in Equation A11 in Vogt13
    A11_coeffs = {0:-4.61777, -1:1.41612, -2:1.52077, -3:-0.63269, -4:0.07386}
    # Calculate relative color excess using Equation A11
    return sum( (A11_coeffs[i] * lambda_um**i) for i in [0,-1,-2,-3,-4] )



def _find_BD(line_lambdas, line_fluxes):
    """
    Given lists or arrays of emission line wavelengths and corresponding fluxes,
    find the Balmer decrement (BD).  We require that fluxes are normalised to
    Hbeta == 1 and we do some checks.
    """
    line_lambdas, line_fluxes = np.array(line_lambdas), np.array(line_fluxes)
    assert line_lambdas.size == line_fluxes.size

    # Find Halpha flux (and Hbeta flux if present):
    rounded_lambdas = np.round(line_lambdas, 0)
    where_Halpha = (rounded_lambdas == 6563) # Halpha is at 6562.819 AA
    where_Hbeta  = (rounded_lambdas == 4861) # Hbeta is at 4861.333 AA
    n_where_Halpha, n_where_Hbeta = np.sum(where_Halpha), np.sum(where_Hbeta)
    if n_where_Halpha == 0:
        raise ValueError("Input lines do not include Halpha (6563 AA)")
    elif n_where_Halpha == 1:
        halpha_flux = line_fluxes[where_Halpha][0] # Scalar (not array)
    else:
        raise ValueError("Multiple lines at a wavelength similar to Halpha!")
    if n_where_Hbeta == 0:
        pass # Not including Hbeta is fine.
    elif n_where_Hbeta == 1:
        if not np.isclose(line_fluxes[where_Hbeta], 1.0, atol=1e-4):
            raise ValueError("Fluxes must be normalised so that Hbeta == 1.0")
    else:
        raise ValueError("Multiple lines at a wavelength similar to Hbeta!")

    # The Balmer decrement F_Halpha / F_Hbeta is equal to F_Halpha since
    # F_Hbeta is normalised to 1.
    return halpha_flux # Return Balmer decrement



def _deredden_BD(line_lambdas, line_fluxes, BD=2.85):
    """
    De-redden emission line fluxes by considering a target intrinsic Balmer
    decrement (BD), using Equation A12 in the Appendix of Vogt13.  Also
    calculate the extinction A_v using Equation A14 in Vogt13.  This is the
    inverse function of "_redden_BD".
    line_lambdas, line_fluxes: As in the function "deredden".
    BD: Desired Balmer decrement (F_Halpha / F_Hbeta) to use in de-reddening
        (will be the Balmer decrement in the output).

    Returns dered_fluxes, an array of de-reddened fluxes corresponding to the
    input line_lambdas. The de-reddened fluxes are normalised to Hb == 1.
    """
    line_lambdas, line_fluxes = np.array(line_lambdas), np.array(line_fluxes)
    assert line_lambdas.size == line_fluxes.size
    obs_BD = _find_BD(line_lambdas, line_fluxes) # Observed Balmer decrement

    # Apply Equation A12 in Vogt13
    r_c_e = _calc_relative_colour_excess(line_lambdas) # Vector of RCE
    dered_fluxes = line_fluxes * (obs_BD / BD)**(0.76*(r_c_e + 4.5))

    # Calculate Hbeta flux and normalise results
    r_c_e_Hb = _calc_relative_colour_excess(4861.333) # Hbeta relative colour excess
    hbeta_dered_flux = (obs_BD / BD)**(0.76*(r_c_e_Hb + 4.5))
    dered_fluxes /= hbeta_dered_flux
    
    return dered_fluxes



def _redden_BD(line_lambdas, line_fluxes, BD):
    """
    Redden emission line fluxes by considering a target Balmer decrement (BD),
    using an inverted version of Equation A12 in the Appendix of Vogt13.  Also
    calculate the extinction A_v that is being "added", using Equation A14 in
    Vogt13.  This is the inverse function of "_deredden_BD".
    line_lambdas, line_fluxes: As in the function "redden".
    BD:    Desired Balmer decrement (F_Halpha / F_Hbeta) to use for reddening
           (will be the Balmer decrement in the output).

    Returns red_fluxes, an array of reddened fluxes corresponding to the input
    line_lambdas. The reddened fluxes are normalised to Hb == 1.
    """
    line_lambdas, unred_fluxes = np.array(line_lambdas), np.array(line_fluxes)
    assert line_lambdas.size == unred_fluxes.size
    intrinsic_BD = _find_BD(line_lambdas, unred_fluxes)
    if BD < intrinsic_BD: # BD == intrinsic_BD is okay.
        raise ValueError("The target Balmer decrement must not be less than the"
                         "input Balmer decrement")

    # Apply inverted form of Equation A12 in Vogt13
    r_c_e = _calc_relative_colour_excess(line_lambdas) # Vector of RCE
    red_fluxes = unred_fluxes * (intrinsic_BD / BD)**(0.76*(r_c_e + 4.5))

    # Calculate Hbeta flux and normalise output results
    r_c_e_Hb = _calc_relative_colour_excess(4861.333) # Hbeta relative colour excess
    hbeta_dered_flux = (intrinsic_BD / BD)**(0.76*(r_c_e_Hb + 4.5))
    red_fluxes /= hbeta_dered_flux

    return red_fluxes



def _BD_from_Av_for_dereddening(line_lambdas, line_fluxes, A_v):
    """
    Find the dereddened Balmer decrement (BD) that would arise from "removing"
    an extinction of A_v (magnitudes) from the line_fluxes.
    line_lambdas, line_fluxes: As in the function "deredden".
    A_v: The extinction (magnitudes).

    Returns the Balmer decrement dereddened_BD (F_Halpha / F_Hbeta).
    """
    assert A_v >= 0
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
    A_v: The extinction (magnitudes).

    Returns the Balmer decrement reddened_BD (F_Halpha / F_Hbeta).
    """
    assert A_v >= 0
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

    Returns A_v, the extinction in magnitudes as a float.
    """
    assert BD_low <= BD_high
    A_v = 8.55 * np.log10( BD_high / BD_low ) # Equation A14 in Vogt13
    return A_v



def deredden(line_lambdas, line_fluxes, line_errs=None, BD=None, A_v=None):
    """
    Deredden emission line fluxes by either specifying a target Balmer decrement
    (FHalpha/FHbeta), or an extinction A_v to be "removed".
    This is the inverse function of "redden".
    line_lambdas: Float, or list/array of floats.  Wavelengths in Angstroems.
    line_fluxes:  Corresponding float, list or array of reddened fluxes.
                  Fluxes are relative to Hbeta (4861.3 A) == 1.  Note that
                  line_lambdas and line_fluxes must include Halpha (6562.8 A).
    line_errs:    Corresponding quantities to be scaled with each flux.
    A_v: The assumed extinction (magnitudes) to "remove" when dereddening.
    BD:  The target (intrinsic) Balmer decrement.
    Only one of "A_v" or "BD" may be specified.  If neither is set, BD=2.85 is
    used.

    Returns dered_fluxes if line_errs is not specified, or alternatively returns
    (dered_fluxes, dered_errs) is line_errs is given.
    The array dered_fluxes contains de-reddened fluxes corresponding to the
    input line_lambdas.  The dereddened fluxes are normalised to Hbeta == 1. The
    array dered_errs is line_errs with the same element-wise scaling as dered_fluxes.
    """
    # Look at what was specified, and determine the balmer decrement to use.
    if A_v is None:
        if BD is None:
            BD = 2.85
    else: # A_v is specified
        if BD is not None:
            raise ValueError("Must specify only one of A_v or BD, not both")
        BD = _BD_from_Av_for_dereddening(line_lambdas, line_fluxes, A_v)       
    dered_fluxes = _deredden_BD(line_lambdas, line_fluxes, BD)

    if line_errs is None:
        return dered_fluxes
    else:
        line_errs = np.array(line_errs)
        assert line_errs.size == len(line_lambdas) # line_lambdas may be list
        dered_errs = line_errs * (dered_fluxes / line_fluxes)
        return dered_fluxes, dered_errs



def redden(line_lambdas, line_fluxes, line_errs=None, BD=None, A_v=None):
    """
    Redden emission line fluxes by either specifying a target Balmer decrement
    (FHalpha/FHbeta), or an extinction A_v to be "applied".
    This is the inverse function of "deredden".
    line_lambdas: Float, or list/array of floats.  Wavelengths in Angstroems.
    line_fluxes:  Corresponding float, list or array of intrinsic fluxes.
                  Fluxes are relative to Hbeta (4861.3 A) == 1.  Note that
                  line_lambdas and line_fluxes must include Halpha (6562.8 A).
    line_errs:    Corresponding quantities to be scaled with each flux.
    A_v: The assumed extinction (magnitudes) to "apply" when reddening.
    BD:  The target Balmer decrement.
    Only one of "A_v" or "BD" may be specified.

    Returns red_fluxes if line_errs is not specified, or alternatively returns
    (red_fluxes, red_errs) is line_errs is given.
    The array red_fluxes contains reddened fluxes corresponding to the input
    line_lambdas.  The dereddened fluxes are normalised to Hbeta == 1.  The
    array red_errs is line_errs with the same element-wise scaling as red_fluxes.
    """
    # Look at what was specified, and determine the balmer decrement to use.
    if A_v is None:
        if BD is None:
            raise ValueError("Must specify one of A_v or BD")
    else: # A_v is specified
        if BD is not None:
            raise ValueError("Must specify only one of A_v or BD, not both")
        BD = _BD_from_Av_for_reddening(line_lambdas, line_fluxes, A_v)       
    red_fluxes = _redden_BD(line_lambdas, line_fluxes, BD)

    if line_errs is None:
        return red_fluxes
    else:
        line_errs = np.array(line_errs)
        assert line_errs.size == len(line_lambdas) # line_lambdas may be list
        red_errs = line_errs * (red_fluxes / line_fluxes)
        return red_fluxes, red_errs



def _test():
    # Run some simple tests of the functionality of this module
    atol = 5e-4  # Absolute tolerance for comparing fluxes (Hbeta == 1)
    # Decrease this tolerance to watch the tests fail!
    def test_arrays(arr1, arr2):
        # Test approximate equality of arrays
        return np.all(np.isclose(arr1, arr2, atol=atol))

    # Check that Halpha flux/Balmer decrement behaves as expected
    assert np.isclose(_deredden_BD([6563],[3.5], 2.9)[0], 2.9, atol=atol)
    assert np.isclose(_redden_BD([6563],[2.9], 3.5)[0], 3.5, atol=atol)
    assert np.isclose(deredden([6563],[3.5], BD=2.9)[0], 2.9, atol=atol)
    assert np.isclose(redden([6563],[2.9], BD=3.5)[0], 3.5, atol=atol)

    # Test data
    BD1, BD_intrinsic = 3.41, 2.9
    obs_lambdas = [6563, 6583, 3726.032, 4861.33] # Halpha, [NII], [OII], Hbeta
    obs_fluxes  = [BD1,  4.1,  1.35,     0.99999]

    # Test that "_deredden_BD" and "_redden_BD" are inverse functions
    dered_fluxes_1 = _deredden_BD(obs_lambdas, obs_fluxes, BD=BD_intrinsic)
    rered_fluxes_1 = _redden_BD(obs_lambdas, dered_fluxes_1, BD=BD1)
    assert test_arrays(rered_fluxes_1, obs_fluxes)

    # Test that "_BD_from_Av_for_dereddening", "_BD_from_Av_for_dereddening"
    # and "Av_from_BD" are consistent
    Av1 = Av_from_BD(BD_low=BD_intrinsic, BD_high=BD1)
    BD_a = _BD_from_Av_for_dereddening(obs_lambdas, obs_fluxes, A_v=Av1)
    assert np.isclose(BD_a, BD_intrinsic, atol=atol)
    BD_b = _BD_from_Av_for_reddening(obs_lambdas, dered_fluxes_1, A_v=Av1)
    assert np.isclose(BD_b, BD1, atol=atol)

    # Test that "deredden" and "redden" are inverse functions, using A_v
    dered_fluxes_2 = deredden(obs_lambdas, obs_fluxes, A_v=Av1)
    assert test_arrays(dered_fluxes_2, dered_fluxes_1)
    rered_fluxes_2 = redden(obs_lambdas, dered_fluxes_2, A_v=Av1)
    assert test_arrays(rered_fluxes_2, rered_fluxes_1)
    assert test_arrays(rered_fluxes_2, obs_fluxes)

    # Test that an extinction of 0 results in negligible flux change
    same_fluxes = redden(obs_lambdas, obs_fluxes, A_v=0)
    assert test_arrays(same_fluxes, obs_fluxes)

    print("Tests passed :D")



if __name__ == "__main__":
    # If we run this module as a script, do some tests
    _test()

