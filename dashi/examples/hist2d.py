
import numpy as n
import dashi as d
import pylab as p

d.visual()

x = n.random.normal(2, 2, 1e5)
y = n.random.normal(-1, 3, 1e5)

h = d.factory.hist2d( (x,y), 100, labels=("x", "y"))

p.figure(figsize=(8,8))

p.subplot(221)
h.imshow()
cb = p.colorbar()
cb.set_label("bin count")

p.subplot(222)
h.imshow(log=1)
cb = p.colorbar()
cb.set_label("log10(bin count)")

p.subplot(223)
h.contour(filled=1)
cb = p.colorbar()
cb.set_label("bin count")

p.subplot(224)
h.contour(filled=0, levels=[.5,1,1.5], log=1,clabels=1)
cb = p.colorbar()
cb.set_label("log(bin count)")
p.xlim(-5,10)
p.ylim(-10,10)

