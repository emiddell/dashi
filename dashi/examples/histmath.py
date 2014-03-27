
import numpy as n
import dashi as d
import pylab as p

d.visual()

bins = n.linspace(-10,10,101)
h1 = d.factory.hist1d(n.random.normal(2,2,1e3), bins)
h2 = d.factory.hist1d(n.random.normal(2,2,1e5), bins)


p.figure(figsize=(8,4))

(h1/1e8).scatter(log=1)
(h2/1e4).scatter(log=1)

