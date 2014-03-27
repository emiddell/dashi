
from dashi.datasets import Variable, DatasetHub, usermethod, HDFDataset
import numpy as n
import dashi as d
import tables
from collections import defaultdict
import os, sys

try:
    from dashi.datasets import HDFChainDataset
except ImportError:
    from nose import SkipTest
    raise SkipTest

# create test hdf files
categories = ["typeA", "typeB", "typeC"]
datafiles = defaultdict(list)
nfiles = 3

for i,name in enumerate(["typeA", "typeB", "typeC"]):
    for j in range(nfiles):
        arr = n.zeros(100, dtype=[("x", float), ("y", int)])
        arr["x"] = n.arange(0, 100, dtype=float)
        arr["y"] = n.arange(100, 0, -1, dtype=int)

        fname = "%s_%d.h5" % (name,j)
        f = tables.openFile(fname, "a")
        f.createTable("/", "test", arr)
        f.close()

        datafiles[name].append(fname)

hub = DatasetHub()
for cat in categories:
    hub.connect_dataset(cat, HDFChainDataset(datafiles[cat], verbose=1))

hub_write = DatasetHub()
for cat in categories:
    fname = "write_%s.h5" % cat
    hub_write.connect_dataset(cat, HDFDataset(fname, "a"))
    datafiles[cat].append(fname)

b = hub.get("/test:x")

print b

assert len(b.typeA) == nfiles*100
assert len(b.typeB) == nfiles*100
assert len(b.typeC) == nfiles*100

hub_write.put("/x", b)

for cat,files in datafiles.iteritems():
    for f in files:
        print "delete", f
        os.remove(f)
