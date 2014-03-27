
import numpy as n
import dashi as d
import pylab as p

d.visual()

data1 = n.random.normal(2,2,1e3)
data2 = n.random.normal(5,4,1e3)
data = n.hstack((data1,data2))
h = d.factory.hist1d( data, 40, label="x")

p.figure(figsize=(8,4))

p.subplot(131)
h.line()

p.subplot(132)
h.scatter()

p.subplot(133)
h.band()
