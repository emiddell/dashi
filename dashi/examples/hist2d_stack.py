"""
 use of 2d histograms for correlation studies
"""
import numpy as n
import dashi as d
import pylab as p

d.visual()

npoints = 1e5
y = n.random.uniform(0,10,npoints)
x = 2 * y + 3 + n.random.normal(0,2,npoints)

h = d.factory.hist2d((x,y),
                     (n.linspace(0,30,101), n.linspace(0,10,11)),
                     labels=("x","y")
                    )

p.figure(figsize=(9,9))
p.subplots_adjust(wspace=.25)

# simple imshow of bincontent array
p.subplot(221)
h.imshow()

# profile plots. project on dimension 1 ("y")
p.subplot(222)
scatterpoints = d.histfuncs.h2profile(h,dim=1)
scatterpoints.scatter()

# stacked 1d histograms plots
# keeping the overall shape of the projected distribution
p.subplot(223)
h.stack1d()

# normalizing each bin to 1 to emphasize relative contributions
p.subplot(224)
h.stack1d(boxify=1)

