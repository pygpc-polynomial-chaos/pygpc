import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
import pygpc
import math
# chebyshev

n = 100

# x_beta = np.random.beta(.5, .5, n)
x = np.linspace(0.01, 0.99, 1000)

def cheb(x):

    f = (np.pi * (x * (1-x))**(1/2))**(-1)
    return f

# beta = scipy.stats.beta(0.5, 0.5,loc=0,scale=1).pdf(x)
# x_cheb = cheb(x)
# plt.subplot(212)
# plt.plot(x, beta)
# plt.subplot(211)
# plt.plot(x, x_cheb)
# plt.show

# x = np.linspace(0.01, 0.99, 10000)
# u = np.random.rand(10000)
# y = np.cos(np.pi*u)
#
# plt.scatter(y, x)
# plt.show

def MCMC_Hampton(n, p, d, dist='uniform'):

    if p > d:
        if dist is 'uniform':

            def w(x):
                w_facts = np.zeros([d])
                for i in range(d):
                    w_facts[i] = (1-x[i]**2)**1/4
                w = np.prod(w_facts)
                return w

            def G(x):
                return scipy.stats.beta(.5, .5, loc=0, scale=1).pdf(x)
            def g(x):
                return scipy.stats.beta.pdf(x, .5, .5)
            def f(x):
                return scipy.stats.beta(1, 1).pdf(x)

    def Metropolis_Hastings(n_):

        n = max(2 * n_, 10000)
        # draw n samples from the proposal distribution
        samples = np.random.beta(.5, .5, size=[n, d])

        # Metropolis-Hastings with 10,000 iterations.
        for i in range(n):
            u = np.random.rand(d)
            # 'iterate' over a number of burn in Samples
            for j in range(10):
                x = samples[i, :]
                x_ = np.random.beta(.5, .5, size=d)
                rho = np.zeros(d)

                for k in range(d):
                    rho[k] = np.min(1, (g(x[k]) * f(x_[k]) * (w(x_) ** (-2))/(g(x_[k]) * f(x[k]) * (w(x) ** (-2)))))
                # draw a uniform sample from [0, 1]
                if (u.sum() < rho.sum()):
                    samples[i, :] = x_
                    u = rho
        samples = samples[len(samples)-n: len(samples)]
        return samples

    return Metropolis_Hastings(n)
n = 1000
set = MCMC_Hampton(n, 8, 4)
mu = pygpc.mutual_coherence(set)/pygpc.mutual_coherence(np.random.beta(1, 1, size=[n, 4]))
print(mu)