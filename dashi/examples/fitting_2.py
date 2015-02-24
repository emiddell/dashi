"""
   fitting a  parabola using dashi.scatterpoints.points2d as 
   a data container
"""
import dashi as d; d.visual()
import numpy as n
import pylab as p

yerr=10.
data = d.scatterpoints.points2d()
data.x = n.linspace(-10,10,21)
data.y = 2*(data.x-2)**2 - 3 + n.random.normal(0,yerr,len(data.x))
data.yerr = yerr *  n.ones(len(data.x)) # initial guess for the error

mod = d.poly(2)
mod = d.fitting.leastsq(data.x,data.y,mod, chi2values=True)

p.figure(figsize=(9,4))
p.subplots_adjust(wspace=.25)

p.subplot(121)
data.scatter(fmt="ko", ms=3)
p.plot(data.x, mod(data.x), "r--")
mod.parbox(loc=1)

ax = p.subplot(122)
p.plot(mod.chi2values[0], mod.chi2values[1], "k-")
p.text(0.1,0.9, "$\chi^2/ndof = %.2f/%d$" % (mod.chi2, mod.ndof), transform = ax.transAxes) 
p.ylabel("chi2 contribution")

