from __future__ import print_function, division
import numpy as np  # Core numerical library
import numpy.random as rand
import emcee
from .bigelm_plotting import plot_MCMC_chain




"""
Adam D. Thomas 2015 - 2016



"""



def calculate_posterior(p_list, Grid_container, Obs_Container, log_prior_func):
    """
    Evaluate the posterior at a particular point in the parameter space.
    p_list:  list of parameter values (a point in the parameter space)
    Grid_container: Has attribute "flux_interpolators", which are functions that
       return the interpolated emission-line flux for a set of parameter values.
    Obs_Container: Has attributes lines_list, obs_fluxes, and obs_flux_errors
    log_prior: Value of the logarithm of the prior at the point p_list

    Returns the natural log of the posterior at the point p_list.
    """
    # Firstly: we return a very negative posterior if we're outside the
    # parameter space, more negative if we're further away
    p_out_list = []
    for p, (p_min, p_max) in zip(p_list, Grid_container.Raw_grids.p_minmax):
        dist_min, dist_max = p - p_min, p_max - p # Both > 0 if p inside bounds
        if dist_min < 0 or dist_max < 0:
            range_p = p_max - p_min
            p_out_list.append( -min(dist_min, dist_max) / range_p ) # Normalised
    if len(p_out_list) > 0:
        tot_range = np.sqrt(np.sum([d**2 for d in p_out_list]))
        return -10**(150 + tot_range)
        # Return increasingly small posterior the further we are away from
        # the allowed region

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
        # Use a log version of equation 3 on pg 4 of Blanc et al. 2015 (IZI)
        # N.B. var is the sum of variances due to both the measured and modelled fluxes
        pred_flux_i = Grid_container.flux_interpolators[emission_line](p_list)
        pred_flux_i = max(0, pred_flux_i) # Ensure nonnegative - inteprolation may be weird...
        var = obs_flux_errors[i]**2 + (epsilon_2 * pred_flux_i)**2
        log_likelihood += ( - (( obs_fluxes[i] - pred_flux_i )**2 / 
                                 ( 2.0 * var ))  -  0.5*np.log( var ) )
        # N.B. "log" is base e
    log_likelihood += np.log(2 * np.pi)

    # Note: ??????? The parameter space, space of measurements and space of predictions are all continuous.
    # Each value P in the posterior array is differential, i.e. P*dx = (posterior)*dx
    # for a vector of parameters x.  # ????????????????

    log_prior = log_prior_func(Grid_container, p_list)
    # Return (un-normalised) log posterior (product of likelihood and prior):
    return log_likelihood + log_prior  # Posterior from Bayes' Theorem!



def uniform_prior(Grid_container, p_list):
    """
    Return the natural logarithm of a uniform prior.
    Grid_container: Contains details of the input model grid
    p_list: List of parameter values at a point in the parameter space
    Returns the log of the value of the prior at the point p_list.
    Since this is a uniform prior, p_list is ignored.
    """
    ranges = [(a.max() - a.min()) for a in Grid_container.Raw_grids.val_arrs]
    prior = 1. / np.product( ranges )
    return np.log( prior )



# class MH_proposal_axisaligned_constrained(object):
#     """
#     A Metropolis-Hastings proposal, with axis-aligned Gaussian steps,
#     for convenient use as the ``mh_proposal`` option to
#     :func:`EnsembleSampler.sample` .
#     ADT: I added a feature to not allow going outside the given params ranges
#     """
#     # Modified from MH_proposal_axisaligned in
#     # /Applications/anaconda/lib/python3.5/site-packages/emcee/utils.py
#     def __init__(self, stdev, min_X, max_X):
#         self.stdev = stdev # List the standard deviation to use in each dimension
#         self.min_X = min_X # List min allowed value for each parameter
#         self.max_X = max_X # List max allowed value for each parameter
#         assert len(stdev) == len(min_X) and len(stdev) == len(max_X)
#         assert np.all(max_X > min_X)

#     def __call__(self, X):
#         (nw, npar) = X.shape # Number of walkers and the dimension of the space
#         assert(len(self.stdev) == npar)
#         new_X = X + self.stdev * np.random.normal(size=X.shape)
#         low_diff = new_X - self.min_X # Should all be positive
#         high_diff = new_X - self.max_X # Should all be negative
#         if np.all(low_diff > 0) and np.all(high_diff < 0):
#             return new_X
#         # If the proposed values aren't in the allowed ranges, let's fix them:
#         ranges = (self.max_X - self.min_X)
#         where_low, where_high = low_diff < 0, low_diff > 0
#         new_X[where_low] = np.amin(new_X[where_low]-2*low_diff[where_low])


#         return X + self.stdev * np.random.normal(size=X.shape)




def fit_MCMC(Grid_container, Obs_Container, nwalkers, nburn, niter,
                                            burnchainplot=None, chainplot=None):
    """
    Run MCMC fitting
    Grid_container: Contains details of the input model grid
    Obs_Container: Has attributes lines_list, obs_fluxes, and obs_flux_errors,
                   and is passed through to the function "calculate_posterior"
    nwalkers: number of "walkers" to use in exploring the parameter space
    nburn: number of burn-in iterations to discard at the start
    niter: number of iterations to perform after completing "burn-in"

    Returns the emcee.EnsembleSampler object.
    """
    # Note: used starmodel.py in "isochrones" by Tim D Morton for inspiration
    print("Running MCMC sampling...")

    # Initialise the sampler:
    log_prior_func = uniform_prior # Uniform prior function
    # kwargs to be passed through to "calculate_posterior":
    Raw_grids = Grid_container.Raw_grids
    kwargs = {"Grid_container":Grid_container, "Obs_Container":Obs_Container,
              "log_prior_func":log_prior_func}
    sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=Raw_grids.ndim, 
                                    lnpostfn=calculate_posterior, kwargs=kwargs)
    # Docs at 
    # http://dan.iel.fm/emcee/current/api/#the-affine-invariant-ensemble-sampler

    # Determine the starting distribution:
    centre = [(a.max() - a.min())/2.0 for a in Raw_grids.val_arrs]
    # centre is a vector in the middle of the parameter space
    ranges = [(a.max() - a.min()) for a in Raw_grids.val_arrs]
    # ranges is the range of each parameter
    p0_lists = []
    for i in range(Raw_grids.ndim):  # Iterate over parameters
        p0_list_i = rand.normal(size=nwalkers, scale=ranges[i]/1e4) + centre[i]
        p0_lists.append( p0_list_i.tolist() )
    p0 = np.array(p0_lists).T # shape (nwalkers, ndim)
    # Now each row of p0 contains the starting coordinates for a walker,
    # and each column corresponds to a parameter.
    # We've set an N-D normal distribution in the middle of the parameter
    # space for p0.  Is this any good?
    
    # Run the sampler!
    # Burn in:
    for i, (pos,lnprob,rstate) in enumerate(sampler.sample(p0, iterations=nburn)):
        print("Step", i, "({0:5.1%})".format(i*1.0/nburn*100))
    # Have a look at the burn-in
    param_display_names = list(Grid_container.Params.display_names.values())
    plot_MCMC_chain(sampler.chain, param_display_names, show=True, outfile=burnchainplot)
    
    sampler.reset()
    # Now run for real:
    print("Finished burn-in, running for real...")
    for i, (pos,lnprob,rstate) in enumerate(sampler.sample(pos, iterations=niter,
                                                                rstate0=rstate)):
        print("Step", i, "({0:5.1%})".format(i*1.0/niter*100))
    # Have a look at the final results
    plot_MCMC_chain(sampler.chain, param_display_names, show=True, outfile=chainplot)

    # # Burn in
    # pos, lnprob, rstate = sampler.run_mcmc(p0, nburn)
    # sampler.reset()
    # # Now run for real:
    # sampler.run_mcmc(pos, niter, rstate0=rstate)

    print("MCMC sampling finished")
    return sampler
    



