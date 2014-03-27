"""
   fitting a straight line
"""
import dashi as d; d.visual()
import numpy as n
import pylab as p

x = n.linspace(-10,10,21)
y = 2*x + 3 + n.random.normal(0,1,len(x))

mod = d.poly(1)
mod = d.fitting.leastsq(x,y,mod)
p.errorbar(x,y,yerr=1,fmt="ko",linestyle="none", capsize=0)
p.plot(x, mod(x), "r--")
mod.parbox(loc=2)
