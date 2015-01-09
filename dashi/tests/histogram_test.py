
import numpy as n
import dashi as d
from nose.tools import *
from nose import SkipTest
import os


def test_abstract_histogram_creation():
    nbins = 10
    hist = d.histogram.histogram(1, (n.linspace(-10,10,nbins+1),) )

def test_1d_histogram_creation():
    nbins = 10
    hist = d.histogram.hist1d(n.linspace(-10,10,nbins+1))

def test_2d_histogram_creation():
    nbins = 10
    hist = d.histogram.hist2d((n.linspace(-10,10,nbins+1), n.linspace(-5,5,nbins+1)))


def test_abstract_histogram_binning():
    nbins = 10
    hist = d.histogram.histogram(1, (n.linspace(-10,10,nbins+1),) )
    eps = 1e-14

    assert len(hist._h_bincontent) == nbins+2
    assert hist._h_binedges[0][0] == - n.inf
    assert hist._h_binedges[0][-1] == + n.inf

    assert ((hist._h_binedges[0][1:-2] + hist._h_binwidths[0] - hist._h_binedges[0][2:-1]) < eps).all()
    assert ((hist._h_binedges[0][1:-2] + 0.5*hist._h_binwidths[0] - hist._h_bincenters[0]) < eps).all()

def test_1d_histogram_binning():
    nbins = 10
    hist = d.histogram.hist1d(n.linspace(-10,10,nbins+1))
    eps = 1e-14

    assert len(hist.bincontent) == nbins
    assert hist.binedges[0] != - n.inf
    assert hist.binedges[-1] !=  n.inf

    assert ((hist.binedges[:-1] + hist.binwidths - hist.binedges[1:]) < eps).all()
    assert ((hist.binedges[:-1] + 0.5*hist.binwidths - hist.bincenters) < eps).all()

    
def test_1d_fillprotection():
    nbins = 10
    hist = d.histogram.hist1d(n.linspace(-10,10,nbins+1))
    assert_raises(ValueError, hist.fill, [1], [n.nan]) 


def test_1d_fill_nans():
    nbins = 10
    hist = d.histogram.hist1d(n.linspace(-10,10,nbins+1))
    hist.fill( n.array([1, 1, 2, n.nan, 0, n.nan]))

    assert hist.stats.nans == 2
    assert hist.stats.nans_wgt == 2
    assert hist.stats.nans_sqwgt == 2
    assert hist.stats.nentries == 6
    assert hist.stats.weightsum == 6
    assert hist.bincontent.sum() == 4

def test_1d_fill_nans_with_weights():
    nbins = 10
    hist = d.histogram.hist1d(n.linspace(-10,10,nbins+1))
    sample  = n.array([1, 1, 2, n.nan, 0, n.nan])
    weights = n.array([2, 2, 3, 5,     1, 3])
    hist.fill( sample, weights)

    assert hist.stats.nans == 2
    assert hist.stats.nans_wgt == 8
    assert hist.stats.nans_sqwgt == (25 + 9)
    assert hist.stats.nentries == 6
    assert hist.stats.weightsum  == 2+2+3+5+1+3
    assert hist.bincontent.sum() == 2+2+3  +1

def test_2d_fill_nans():
    nbins = 10
    hist = d.histogram.hist2d( (n.linspace(-10, 10,nbins+1), 
                                n.linspace(  0, 5, nbins+1)) )
    hist.fill( (n.array([ 1, 1, 2, n.nan, 0, n.nan]),
                n.array([-1, 2, 3,     4, 1, 2    ])) )

    assert hist.stats.nans == 2
    assert hist.stats.nans_wgt == 2
    assert hist.stats.nans_sqwgt == 2
    assert hist.stats.nentries == 6
    assert hist.stats.weightsum == 6
    assert hist.bincontent.sum() == 3

def test_2d_fill_nans_with_weights():
    nbins = 10
    hist = d.histogram.hist2d( (n.linspace(-10, 10,nbins+1), 
                                n.linspace(  0, 5, nbins+1)) )
    weights   = n.array([ 2, 2, 3,     5, 1, 3])
    
    hist.fill( (n.array([ 1, 1, 2, n.nan, 0, 2]),
                n.array([-1, 2, 3,     4, 1, n.nan])),
                weights)

    assert hist.stats.nans == 2
    assert hist.stats.nans_wgt == 5+3
    assert hist.stats.nans_sqwgt == 25 + 9
    assert hist.stats.nentries == 6
    assert hist.stats.weightsum == 2+2+3+5+1+3
    assert hist.bincontent.sum() == 2+3+1

## some usage_tests:

def with_all_minimizers(f):
    def test():
        if d.fitting._minimize is None:
            raise SkipTest
        mini = d.fitting._minimize
        for m in d.fitting._minimizers:
            d.fitting._minimize = m
            f()
        d.fitting._minimize = mini
    return test

@with_all_minimizers
def test_fitting():    
    # reproducibility considered good
    n.random.seed(1)
    nbins = 50
    bins = n.linspace(-10,10, nbins+1)
    sample = n.random.normal(2, 0.2, 1e5)
    weights = n.random.normal(1, 0.3, 1e5)
    weights[weights < 0] = 0
    hist = d.histogram.hist1d(bins)
    hist.fill(sample, weights)
    
    mod1 = hist.leastsq(d.gaussian(), verbose=False)
    assert(abs(mod1.params["mean"] - 2)/2 < 1e-3)
    assert(abs(mod1.params["sigma"] - .2)/.2 < 2e-1)

@with_all_minimizers
def test_fitting_powerlaw():
    # reproducibility considered good
    n.random.seed(1)
    # sample from a power law by inversion
    index = -2.7
    xmin, xmax = 1e2, 1e4
    g = index + 1
    sample = (xmin**g + n.random.uniform(size=1e4)*(xmax**g - xmin**g))**(1./g)
    bins = n.logspace(2, 4, 101)
    hist = d.factory.hist1d(sample, bins=bins)
    
    # fit once with a linear 
    mod = hist.leastsq(d.fitting.powerlaw())
    assert(abs((mod.params["index"]-index)/index) < 2e-2)
    
    mod = hist.poissonllh(d.fitting.powerlaw())
    assert(abs((mod.params["index"]-index)/index) < 2e-2)
    
def test_plotting():
    import matplotlib as mpl
    mpl.use("agg")

    import pylab as p
    d.visual()
    nbins = 50
    bins = n.linspace(-10,10, nbins+1)
    sample = n.random.normal(2, 0.2, 1e5)
    weights = n.random.normal(1, 0.3, 1e5)
    weights[weights < 0] = 0
    hist = d.histogram.hist1d(bins)
    hist.fill(sample, weights)

    hist.scatter()
    #mod = hist.leastsq(d.gaussian(), verbose=False)
    #p.plot(hist.bincenters, mod(hist.bincenters))
    #hist.statbox(loc=1)
    #mod.parbox(loc=2)
    fname = "/tmp/testfigure.png"
    p.savefig(fname)
    assert os.path.exists(fname)
    os.remove(fname)

def test_cumulative_bincontent():
    """
    Ensure that the cumulative sum over visible bins and over/underflow is
    equivalent to a cumulative sum over the entire histogram backing array.
    """
    def verify_cumsum(h):
        for op in '<', '>':
            for kind in 'bincontent', 'binerror':
                func = lambda arr, axis: d.histfuncs.cumsum(arr, operator=op, axis=axis)
                if kind == 'bincontent':
                    cum = d.histfuncs.cumulative_bincontent(h, op)
                    cum_full = n.apply_over_axes(func, h._h_bincontent, range(h.ndim-1, -1, -1))[h._h_visiblerange]
                else:
                    cum = d.histfuncs.cumulative_binerror(h, op)
                    cum_full = n.sqrt(n.apply_over_axes(func, h._h_squaredweights, range(h.ndim-1, -1, -1))[h._h_visiblerange])
                assert((cum == cum_full).all())
                # assert(False)
    
    for ndim in range(1, 5):
        bins = [n.linspace(0, 1, i+3) for i in range(ndim)]
        if ndim == 1:
            h = d.histogram.hist1d(bins[0])
        elif ndim == 2:
            h = d.histogram.hist2d(bins)
        else:
            h = d.histogram.histogram(ndim, bins)
        
        sample = tuple((n.random.uniform(-1, 2, size=100) for i in range(ndim)))
        h.fill(sample)
        verify_cumsum(h)

def test_rebin():
    for ndim in range(1, 5):
        bins = [n.linspace(0, 1, i+10) for i in range(ndim)]
        h = d.histogram.create(ndim, bins)
        
        sample = tuple((n.random.uniform(-1, 2, size=10000) for i in range(ndim)))
        h.fill(sample)
        
        for axis in range(ndim):
            hb = h.rebin_axis(axis, 2)
            assert(hb._h_bincontent.sum() == h._h_bincontent.sum())
            assert(hb._h_squaredweights.sum() == h._h_squaredweights.sum())

    #h2 = h.hist2d(...)
#    h2.profile(axis, method="stdev") # or "gaussian" axis=0,1, maybe define methods profilex y 
#    h2.scatter()
#    h2.contour()
#    h2.contourf()
#    h2.imshow()
#    h2.serialize(hdfgroup)


    #h3.... tuesday.


#def test_usage():
#    samplesize = 1e5
#    sample1 = n.random.normal(0,  1, samplesize)
#    sample2 = n.random.normal(3, .2, samplesize)
#    weights = n.random.exponential(.2, samplesize) 
#
#    #generation
#    h1 = h.hist1d(sample1, 100)
#    h1 = h.hist1d(sample1, 100, weights)
#    h1 = h.hist1d(sample1, n.linspace(-10,10,101), weights)
#
#
#    assert_raises(ValueError, h.hist1d, None, 100)
#    h1 = h.hist1d(None, n.linspace(-10,10,101))
#    h1.fill(sample1)
#    h1.fill(sample2, weights)
#    h1b = h.emptylike(h1)
#
#    hv.scatter(h1)
#    #hv.scatter(h1, axes=p.gca(), **kwargs)
#    hv.line(h1, filled=True, ec="b", fc="r")
#    hv.band(h1, ec="b", fc="r")
#
#    h2 = h.hist2d((sample1, sample2), 100)
#    h2 = h.hist2d((sample1, sample2), 100, weights)
#    h2 = h.hist2d((sample1, sample2), (n.linspace(-10,10,101), n.linspace(-5,5,21)), weights)
#    h2b = h.emptylike(h2)
#    h2b.fill((sample1, sample2), weights)
#    h3 = h2 + h2b
#
#    hv.profile(h2, axis, method="stdev") # or "gaussian" axis=0,1, maybe define methods profilex y 
#    h2.scatter()
#    h2.contour()
#    h2.contourf()
#    h2.imshow()
