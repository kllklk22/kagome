# kagome

adaptive markov chain monte carlo (mcmc) sampler. 

standard metropolis-hastings algorithms get stuck in local optima if your proposal variance is garbage. this script dynamically adapts the covariance matrix during the burn-in phase so it actually converges on high-dimensional posterior distributions.

used for bayesian inference when your data is too noisy for standard maximum likelihood estimation.
