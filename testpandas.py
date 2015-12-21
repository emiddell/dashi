import numpy as n
import dashi
import dashi.datasets.hub as hub
import dashi.datasets.hdf_datasets as hdf
from dashi.datasets.variable import Variable
signal = hdf.HDFDataset('../testdata/nue_signal.h5')
background = hdf.HDFDataset('../testdata/background.h5')
hu = hub.DatasetHub()
hu.connect_dataset('signal',signal)
hu.connect_dataset('background',background)
#hu.print_toc()
hu.get_vars(vardef_only=True)
hu.vars['energy'] = Variable('/CredoFit:energy', n.linspace(0,6,20))
hu.vars['x'] = Variable('/CredoFit:x', n.linspace(0,6,20))
thevars = hu.get_vars(vardef_only=1)
print thevars
data = hu.get(thevars)
print data
print type(data)
data.x.background
type(data.x.background)
dashi.visual()
hist = dashi.bundleize(dashi.histfactory.hist1d)
dahist = hist(data.energy,n.linspace(0,7,20))
print dahist.keys()

for i in dahist.keys():
    dahist.__getattribute__.line()

#dahist.line()
dahist.background.is_compatible(dahist.signal)
import pylab as p


p.savefig("tescht.png")
