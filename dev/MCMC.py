import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import pygpc
import math
import time

# chebyshev

n = 100

# x_beta = np.random.beta(.5, .5, n)
x = np.linspace(0.01, 0.99, 1000)

def cheb(x):

    f = (np.pi * (x * (1-x))**(1/2))**(-1)
    return f

beta = scipy.stats.beta(0.5, 0.5,loc=0,scale=1).pdf(x)
x_cheb = cheb(x)
plt.subplot(212)
plt.plot(x, beta)
plt.subplot(211)
plt.plot(x, x_cheb)
plt.show

start_time = time.time()

def MCMC_Hampton(n, p, d, dist='uniform'):

    if p > d:
        if dist is 'uniform':

            def w(x):
                w_facts = np.zeros([d])
                for i in range(d):
                    w_facts[i] = (1 - (x[i] ** 2)) ** (1/4)
                w = np.prod(w_facts)
                return w

            def G(x):
                return scipy.stats.beta(.5, .5, loc=0, scale=1).pdf(x)
            def g(x):
                return scipy.stats.beta.pdf(x, .5, .5)
            def f(x):
                return scipy.stats.beta(1, 1).pdf(x)

    def Metropolis_Hastings(n):

        n_burn_in = max(2 * n, 1000)

        # draw n samples from the proposal distribution
        proposal_set = np.random.beta(.5, .5, size=[n_burn_in, d])

        samples = np.zeros([n_burn_in, d])
        i = 1
        # Metropolis-Hastings with 10,000 iterations.
        while i < n_burn_in:
            u = np.random.rand(d)
            # 'iterate' over a number of burn in Samples

            x = proposal_set[i, :]
            x_ = np.random.beta(.5, .5, size=d)
            rho = np.zeros(d)

            for k in range(d):
                a1 = (g(x[k]) * f(x_[k]) * (w(x_) ** (-2)))
                a2 = (g(x_[k]) * f(x[k]) * (w(x) ** (-2)))
                a = a1/a2
                rho[k] = min(1, a)
            # draw a uniform sample from [0, 1]
            if (u.sum() < rho.sum()):
                samples[i, :] = x_
                i = i + 1
        return samples[n:]

    return Metropolis_Hastings(n)
n = 1000
set = MCMC_Hampton(n, 8, 2)
mu = pygpc.mutual_coherence(set)/pygpc.mutual_coherence(np.random.beta(1, 1, size=[n, 4]))
print(set)
print("time: {}seconds " .format(time.time() - start_time))
np.savetxt('MCMC_samples_001.csv', set, delimiter=',')
