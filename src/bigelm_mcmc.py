from __future__ import print_function, division
import numpy as np  # Core numerical library
import numpy.random as rand
import emcee




"""
Adam D. Thomas 2015 - 2016



"""



def calculate_posterior(p_list, Grid_container, Obs_Container, ln_prior):
    """
    Evaluate the posterior at a particular point in the parameter space.
    p_list:  list of parameters
    Grid_container: Has attribute "flux_interpolators", which return the 
                interpolated emission-line flux for a set of parameter values.
    Obs_Container: Has attributes lines_list, obs_fluxes, and obs_flux_errors
    ln_prior: Value of the logarithm of the prior at the point p_list

    Returns the natural log of the posterior at the point p_list.
    """
    print("Calculating posterior at point", p_list)
    # The posterior will have the same shape as an interpolated model grid
    # for one of the emission lines

    # Use systematic uncertainty in modelled fluxes, as in Blanc et al.
    epsilon = 0.15 # dex.  Default is 0.15 dex systematic uncertainty
    # Convert from dex to a scaling factor:
    epsilon_2 = 10**epsilon - 1  # This gives a factor of 0.41 for epsilon=0.15 dex
    # epsilon_2 is a scaling factor to go from a linear modelled flux to an
    # associated systematic error
    # Note the original from izi.pro is equivalent to:
    # epsilon_2 = epsilon * np.log(10)
    # This gives 0.345 for epsilon=0.15 dex. I don't understand this.
    # Why was a log used?  And a log base e?  This is surely wrong.
    # What is intended by this formula anyway?
    # Note that epsilon_2 is the assumed factional systematic error in the model
    # fluxes already normalised to Hbeta.  In izi.pro the default is 0.15 dex,
    # but the default is given as 0.1 dex in the Blanc+15 paper.

    lines_list = Obs_Container.lines_list
    obs_fluxes = Obs_Container.obs_fluxes
    obs_flux_errors = Obs_Container.obs_flux_errors

    # Calculate likelihood:
    log_likelihood = 0  # Initialise likelihood of 1 (multiplictive identity)
    for i, emission_line in enumerate(lines_list):
        # Use a log version of equation 3 on pg 4 of Blanc et al. 2015
        # N.B. var is the sum of variances due to both the measured and modelled fluxes
        pred_flux = Grid_container.flux_interpolators[emission_line][p_list]
        pred_flux = max(0, pred_flux) # Ensure nonnegative - inteprolation may be weird...
        var = obs_flux_errors[i]**2 + (epsilon_2 * pred_flux)**2
        log_likelihood += ( - (( obs_fluxes[i] - pred_flux )**2 / 
                                 ( 2.0 * var ))  -  0.5*np.log( var ) )
        # N.B. "log" is base e

    # Note:  The parameter space, space of measurements and space of predictions are all continuous.
    # Each value P in the posterior array is differential, i.e. P*dx = (posterior)*dx
    # for a vector of parameters x.  # ????????????????

    # Return (un-normalised) log posterior (product of likelihood and prior):
    return log_likelihood + ln_prior  # Posterior from Bayes' Theorem!



def uniform_prior(Grid_container):
    """
    Return the natural logarithm of a uniform prior over the full parameter
    space.  The returned value applies everywhere in the parameter space.
    """
    ranges = [(a.max() - a.min()) for a in Grid_container.val_arrs]
    prior = 1. / np.product( ranges )
    return np.log( prior )



def fit_MCMC(Grid_container, Obs_Container, nwalkers, nburn, niter):
    """
    Run MCMC fitting
    Grid_container: Contain details of the input model grid
    Obs_Container: Has attributes lines_list, obs_fluxes, and obs_flux_errors,
                   and is passed through to the function "calculate_posterior"
    nwalkers: number of "walkers" to use in exploring the parameter space
    nburn: number of burn-in iterations to discard at the start
    niter: number of iterations to perform after completing "burn-in"

    Returns the emcee.EnsembleSampler object.
    """
    # Note: used starmodel.py in "isochrones" by Tim D Morton for inspiration

    # Initialise the sampler:
    ln_prior = uniform_prior(Grid_container) # Calculate uniform prior
    ndim = len(Grid_container.shape) # Dimension of parameter space
    # kwargs to be passed through to "calculate_posterior":
    kwargs = {"Grid_container":Grid_container, "Obs_Container":Obs_Container,
              "ln_prior":ln_prior}
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=ndim, 
                                    lnpostfn=calculate_posterior, kwargs=kwargs)

    # Determine the starting distribution:
    centre = [(a.max() - a.min())/2.0 for a in Grid_container.val_arrs]
    # centre is a vector in the middle of the parameter space
    ranges = [(a.max() - a.min()) for a in Grid_container.val_arrs]
    # ranges is the range of each parameter
    p0_lists = []
    for i in range(ndim):  # Iterate over parameters
        p0_list_i = rand.normal(size=nwalkers, scale=ranges[i]/5.) + centre[i]
        p0_lists.append( p0_list_i.tolist() )
    p0 = np.array(p0_lists).T # shape (nwalkers, ndim)
    # Now each row of p0 contains the starting coordinates for a walker,
    # and each column corresponds to a parameter.
    # We've set an N-D normal distribution in the middle of the parameter
    # space for p0.  Is this any good?
    
    # Run the sampler!
    # Burn in:
    pos, lnprob, rstate = sampler.run_mcmc(p0, nburn)
    sampler.reset()
    # Now run for real:
    sampler.run_mcmc(pos, niter, rstate0=rstate)

    return sampler
    



