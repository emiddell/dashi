
import dashi as d
import numpy, os
from nose import SkipTest

def import_tables():
    try:
        import tables
    except ImportError:
        raise SkipTest
    return tables

def test_save(cleanup=True):
    tables = import_tables()
    fname = 'foo.hdf5'
    h = d.histogram.hist1d(numpy.linspace(0, 1, 11))
    # save to opened file
    with tables.open_file(fname, 'w') as hdf:
        d.histsave(h, hdf, '/', 'foo')
    # let histsave open the file for you
    d.histsave(h, fname, '/', 'foo', overwrite=True)
    if cleanup:
        os.unlink(fname)
    else:
        return fname

def test_load():
    tables = import_tables()
    fname = test_save(False)
    # save to opened file
    with tables.open_file(fname) as hdf:
        d.histload(hdf, '/foo')
    # let histsave open the file for you
    h = d.histload(fname, '/foo')
    assert h.ndim == 1
    assert h.nbins[0] == 10
    os.unlink(fname)
