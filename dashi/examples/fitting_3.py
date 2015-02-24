"""
 example of fitting a more elaborated model (sum of two gaussians)
"""
import numpy as n
import dashi as d
import pylab as p
import scipy.stats
d.visual()


# create random numbers from two normal distributions
x1 = n.random.normal(2,2,1e4)
x2 = n.random.normal(0,6,1e4)

# histogram each
h1 = d.factory.hist1d(x1,n.linspace(-50,20,101))
h2 = d.factory.hist1d(x2,n.linspace(-50,20,101))

# use histogram arithmetic to combine them
h = h1 + h2 

# Define the function that should be fitted to (h1+h2).
# The fist argument is interpreted as the point where 
# the function should be evaluated (more dimensional
# functions are possible by using tuples).
# All other parameters are taken as fit parameters 
# and should get descriptive names.

def twogauss(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    return (amp1 * scipy.stats.norm(loc=mean1, scale=sigma1).pdf(x) +
            amp2 * scipy.stats.norm(loc=mean2, scale=sigma2).pdf(x) )
def twogauss_int(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    return (amp1 * scipy.stats.norm(loc=mean1, scale=sigma1).cdf(x) +
            amp2 * scipy.stats.norm(loc=mean2, scale=sigma2).cdf(x) )

# wrap into a dashi.fitting.model and perform a least square fit
mod = d.fitting.model(twogauss,twogauss_int)
# first guesses
mod.params['amp1'], mod.params['amp2'] = (h.stats.weightsum/2.,)*2
mod.params['mean1'] = -1
mod.params['mean2'] = 1

h.leastsq(mod)


# plot histograms
h.scatter(c="k")
h1.line(c="#777777")
h2.line(c="#555555")
h1.statbox(loc=6)
h2.statbox(loc=3)

# after fitting mod behaves like a normal function with the fitted
# parameters fixed
p.plot(h.bincenters, n.diff(mod.integral(h.binedges)), "r-", linewidth=2,alpha=.5)
mod.parbox(loc=2)

