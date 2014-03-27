"""
    an example to illustate the use of d.histfuncs.histratio
    to compare histograms with each other
"""
import numpy as n
import dashi as d
import pylab as p

d.visual()

# create two histograms of normal distribution
# with shifted means and one histogram having the double number of entries
bins = n.linspace(-10,10,101)
h1a = d.factory.hist1d(n.random.normal(2.0,2,2e5), bins) 
h1b = d.factory.hist1d(n.random.normal(0.0,.5,1e4), bins)
h1 = h1a+h1b
h2 = d.factory.hist1d(n.random.normal(2.0,2,1e5), bins) 

# first figure: just plot the histograms
p.figure(figsize=(8,4))
h1.line(c="r")
h1a.line(c="r", linestyle=":")
h1b.line(c="r", linestyle=":")
h2.line(c="b")
p.axvline(0, color="k", linestyle="dashed")
h1.statbox(loc=1, edgecolor="r")
h2.statbox(loc=2, edgecolor="b")

# second figure:
# calculate the ratio of the two histograms, i.e. 
# f = x/y with errors Dx, Dy
# df = sqrt (DX / y ) **2 + (x/y**2* Dy)**2
p.figure(figsize=(8,4))
d.histfuncs.histratio(h1, h2, log=False, ylabel="h1/h2").scatter(c="k") 
p.axhline(1, color="k", linestyle="dashed")
p.axvline(0, color="k", linestyle="dashed")

# second figure:
# calculate the logarithmic ratio of the two histograms, i.e. 
# f = log10(x/y) with errors Dx, Dy
# df = sqrt( (Dx/x)**2 + (Dy/y)**2 ) / ln(10)
p.figure(figsize=(8,4))
d.histfuncs.histratio(h1, h2, log=True, ylabel="log10(h1/h2)").scatter(c="k")
p.axhline(0, color="k", linestyle="dashed")
p.axvline(0, color="k", linestyle="dashed")


#d.histfuncs.histratio returns an instance of dashi.scatterpoints.points2d
# up to now only one plotting function is defined  
# .scatter uses pylab.errorplot to draw the calculated ratios
