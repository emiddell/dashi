from math import ceil, floor, log10
import numpy as n
from . import scatterpoints
import dashi as d
from logging import getLogger
import collections


def cumsum(narray, operator='<', axis=None):
    """calculates n.cumsum of narray for cuts depending on the operator ('<' or '>') as e.g. E < 10 or E > 10"""
    if operator == '<':
        return n.cumsum(narray, axis=axis)
    elif operator == '>':
        if axis is None:
            idx = slice(None,None,-1)
        else:
            idx = [slice(None)]*narray.ndim
            idx[axis] = slice(None,None,-1)
        return (n.cumsum(narray[idx], axis=axis))[idx]
    else:
        raise ValueError("Non valid operator chosen! Please choose '<' or '>'")

def _cumsum_with_overflow(bincontent, overflow, func):
    """
    Emulate the result of a cumulative sum over the entire histogram backing
    array given only disjoint views into the visible *bincontent* and the
    *overflow* (a slice in each dimension).
    """
    cum = bincontent
    ndim = bincontent.ndim
    # TODO: take axes to sum over as a parameter
    axes = range(ndim-1, -1, -1)
    for i, axis in enumerate(axes):
        # overflow should be a slab with one trivial dimension
        oflow = overflow[axis]
        assert(oflow.ndim == cum.ndim)
        assert(oflow.shape[axis] == 1)
    
        # sum over the visible part of the array
        cum = func(cum, axis=axis)
        # apply all the sums taken so far to the overflow slab, then add only
        # the part that is either in the overflow bin for this dimension, or
        # in a visible bin of another dimension
        idx = [slice(1,-1)]*ndim
        idx[axis] = slice(0,1)
        cum += n.apply_over_axes(func, oflow, axes[:i])[idx]
        
    return cum

def cumulative_bincontent(histogram, operator='<'):
    """
    Return the cumulative sum of bin entries in the histogram.
    
    :param operator: either '<' (sum from the left) or '>' (sum from the right)
    :returns: an array of the same shape as histogram.bincontent
    """
    func = lambda narray, axis: cumsum(narray, operator=operator, axis=axis)
    if operator == '<':
        return _cumsum_with_overflow(histogram.bincontent, histogram.underflow, func)
    elif operator == '>':
        return _cumsum_with_overflow(histogram.bincontent, histogram.overflow, func)
    else:
        raise ValueError("Non valid operator chosen! Please choose '<' or '>'")

def cumulative_binerror(histogram, operator='<'):
    """
    Return the error on the cumulative sum of bin entries in the histogram.
    
    :param operator: either '<' (sum from the left) or '>' (sum from the right)
    :returns: an array of the same shape as histogram.bincontent
    """
    func = lambda narray, axis: cumsum(narray, operator=operator, axis=axis)
    if operator == '<':
        return n.sqrt(_cumsum_with_overflow(histogram.squaredweights, histogram.underflow_squaredweights, func))
    elif operator == '>':
        return n.sqrt(_cumsum_with_overflow(histogram.squaredweights, histogram.overflow_squaredweights, func))
    else:
        raise ValueError("Non valid operator chosen! Please choose '<' or '>'")
     
def project_bincontent(histogram, dim):
    # sum entries in all other dimensions
    dims = list(range(histogram.ndim))
    dims.remove(dim)
    tmp = histogram.bincontent.sum(axis=tuple(dims))
    assert abs(tmp.sum() - histogram.bincontent.sum() < 1e-6)

    return tmp

def binned_mean_and_variance(bincontent, bincenters):
    wsum = bincontent.sum()
    mean        = (bincontent * bincenters ).sum() / wsum
    meansquared = (bincontent * bincenters**2 ).sum() / wsum
    var = meansquared - mean**2

    return (wsum, mean, var)


def number_format(value, prec=5):
    result = None
    factorexp = 0
    absvalue = abs(value)
    if value != 0 and (absvalue >= 1e6 or absvalue <= 1e-6):
        if value < 1:
            factorexp = -divmod(-log10(absvalue),3)[0]*3
        else:
            factorexp = divmod(log10(absvalue),3)[0]*3
        
        commonfactor = 10**factorexp
        value /= commonfactor
    
    if value<1:
        # construct format string for that precision
        fmt = "".join(["%.", str(prec), "f"])
        result =  fmt % value
    else:
        fmt = "".join(["%.", str(prec), "f"])
        result =  fmt % value

    if factorexp != 0:
        result = "%s\xb71e%d" % (result, factorexp)

    return str(result)
    

def number_error_format(value, error):

    result = None
    factorexp = 0
    
    if error != 0 and (error >= 1e6 or error <= 1e-6):
        if error < 1:
            factorexp = -divmod(-log10(error),3)[0]*3
        else:
            factorexp = divmod(log10(error),3)[0]*3
        
        commonfactor = 10**factorexp
        value /= commonfactor
        error /= commonfactor

    if error<1:
        # digits needed for precision
        if error == 0:
            digits = 1
        else:
            digits = str(int(abs(floor(log10(error)))))

        # construct format string for that precision
        fmt = "".join(["%.", digits, "f \u00B1 %.", digits, "f"])
        result =  fmt % (value,error)
    else:
        # digits needed for precision
        try:
            digits = str(int(abs(ceil(log10(error)))))
            # construct format string for that precision
            fmt = "".join(["%", digits, ".0f \u00B1 %", digits, ".0f"])
            result =  fmt % (value,error)
        except:
            digits = '3'
            fmt = "".join(["%", digits, ".0f \u00B1 %", digits, ".0f"])
            result =  fmt % (value,error)
            

    if factorexp != 0:
        result = "(%s) 1e%d" % (result, factorexp)

    return str(result)


def generatebins_1d_tuple(bintuple):
    if isinstance(bintuple, tuple) and len(bintuple) == 3:
        if not ( (bintuple[1] > bintuple[0]) and bintuple[2] > 0 ):
            raise ValueError("problem with provided bins %s" % str(bintuple))

        binwidth = float(bintuple[1] - bintuple[0]) / float(bintuple[2])
        bins = n.linspace(bintuple[0]-binwidth, bintuple[1]+binwidth, bintuple[2]+3).round(12)
        return bins
    else:
        raise ValueError("bintuple must be a tuple (leftedge, rightedgne, nbins)")


def generatebins_1d(sample, nbins):
    sample = n.asarray(sample)
    if not len(sample.shape) == 1:
        raise ValueError("sample must be an array with shape (1,)")
    finsample = sample[n.isfinite(sample)]

    if len(finsample) == 0:
        return n.linspace(0,1,nbins+1)

    mi,ma = finsample.min(), finsample.max()
    if mi == ma:
        mi -= 1
        ma += 1
    tolerance = 0.005 * (ma - mi)
    return n.linspace(mi-tolerance,ma+tolerance,nbins+1)

def generatebins_nd(ndim, sample, nbins):
    # TODO rework generatebins_nd to use generatebins_1d
        
    # analyse given nbins
    if n.isscalar(nbins):
        nbins = (nbins,) * ndim
    elif isinstance(nbins, tuple):
        if not len(nbins) == ndim:
            raise ValueError("nbins is tuple with length != ndim")
        if not all(map(n.isscalar, nbins)):
            raise ValueError("nbins is tuple containing non scalars")
    else:
        raise ValueError("nbins must be either a scalar or a tuple of lenght ndim.")
    
    # analyse given sample
    if isinstance(sample, tuple): 
        if not len(sample) == ndim:
            raise ValueError("sample is tuple with length != ndim")
        sample = n.vstack(sample).T

    goodmask = n.ones(len(sample), dtype=bool)
    for dim in range(ndim):
        goodmask &= n.isfinite(sample[:,dim])
    
    genbins = []
    sample = sample[goodmask]
    for dim, nb in enumerate(nbins):
        slice = sample[:,dim]
        mi,ma = slice.min(), slice.max()
        tolerance = 0.005 * (ma - mi)

        genbins.append( n.linspace(mi-tolerance,ma+tolerance,nb+1) )

    return tuple(genbins)

def h2profile(h2, dim=0, yerror='std'):
    """
        creates the profile plot of a 2d histogram
        h2  : 2d histogram
        dim : dimension to project on
    """

    npoints = len(h2.bincenters[dim])
    yval = n.zeros(npoints, dtype=float)
    yerr = n.zeros(npoints, dtype=float)


    otherdim = None
    if dim == 0:
        otherdim = 1
        getLogger("dashi.h2profile").info("projecting along dimension 0")
        for i in range(npoints):
            wsum, mean, var = binned_mean_and_variance(h2.bincontent[i,:], h2.bincenters[1])
            yval[i] = mean
            if yerror=='std':
                yerr[i] = n.sqrt(var)
            elif yerror=='emean':
                yerr[i] = n.sqrt(var)/n.sqrt(wsum)
            elif yerror=="median":
                medianidx = (h2.bincontent[i,:].cumsum()/h2._h_bincontent[i,:].sum()).searchsorted(.5)
                if (0 <= medianidx) and (medianidx < len(h2.bincenters[1])):
                    yval[i] = h2.bincenters[1][medianidx]
                else:
                    yval[i] = n.nan
                    
                yerr[i] = 0.
            else:
                raise ValueError('yerror must be std (for standard deviation) or emean (for error of mean value)')
    elif dim == 1:
        otherdim = 0
        getLogger("dashi.h2profile").info("projecting along dimension 1")
        for i in range(npoints):
            wsum, mean, var = binned_mean_and_variance(h2.bincontent[:,i], h2.bincenters[0])
            yval[i] = mean
            if yerror=='std':
                yerr[i] = n.sqrt(var)
            elif yerror=='emean':
                yerr[i] = n.sqrt(var)/n.sqrt(wsum)
            elif yerror=="median":
                medianidx = (h2.bincontent[:,i].cumsum()/h2._h_bincontent[:,i].sum()).searchsorted(.5)
                if (0 <= medianidx) and (medianidx < len(h2.bincenters[0])):
                    yval[i] = h2.bincenters[0][medianidx]
                else:
                    yval[i] = n.nan
                yerr[i] = 0.
            else:
                raise ValueError('yerror must be std (for standard deviation) or emean (for error of mean value)')
    else:
        raise ValueError("dim must be either 0 or 1")

    # remove nans and infs (empty bins)
    mask = n.isfinite(yval) & n.isfinite(yerr) 
    prof = scatterpoints.points2d(len(mask[mask]))


    prof.x    = h2.bincenters[dim][mask]
    prof.xerr = h2._h_binwidths[dim][mask]/2. 
    prof.y    = yval[mask]
    prof.yerr = yerr[mask]
    prof.labels[0] = h2.labels[dim]
    prof.labels[1] = h2.labels[otherdim]

    return prof

def histratio(h1, h2, log=False, ylabel="ratio", mask_infs=True):
    # linear case:
    # f = x/y with errors Dx, Dy
    # df = sqrt (DX / y ) **2 + (x/y**2* Dy)**2
    # logarithmic case:
    # f = log10(x/y) with errors Dx, Dy
    # df = sqrt( (Dx/x)**2 + (Dy/y)**2 ) / ln(10)
    df_lin = lambda x,xerr,y,yerr:  n.sqrt( (xerr/y)**2 + (x*yerr/(y**2))**2)
    df_log = lambda x,xerr,y,yerr: n.sqrt( (xerr/x)**2 + (yerr/y)**2 ) / n.log(10)


    if not h1.is_compatible(h2):
        raise ValueError("histograms are not compatible")

    if log:
        ratio = n.log10( h1.bincontent / h2.bincontent )
        errors = df_log(h1.bincontent, h1.binerror, h2.bincontent, h2.binerror)
    else:
        ratio = h1.bincontent / h2.bincontent
        errors = df_lin(h1.bincontent, h1.binerror, h2.bincontent, h2.binerror)

    mask = n.ones(shape=h1.bincontent.shape, dtype=bool)
    if mask_infs:
        mask &= n.isfinite(ratio)
        mask &= n.isfinite(errors)

    prof = None
    if h1.ndim == 1:
        prof = scatterpoints.points2d(len(mask[mask]))
        prof.x[:]    = h1.bincenters[mask]
        prof.xerr[:] = h1.binwidths[mask] / 2.
        prof.y[:]    = ratio[mask]
        prof.yerr[:] = errors[mask]
        prof.labels[0] = h1.labels[0]
        prof.labels[1] = ylabel
        return prof
    elif h1.ndim == 2:
        prof = scatterpoints.grid2d( (len(h1.bincenters[0]), len(h1.bincenters[1])) )
        imask = n.logical_not(mask)
        ratio[imask] = n.nan
        errors[imask] = n.nan
        prof.x    = h1.bincenters[0] 
        prof.xerr = h1.xerr
        prof.y    = h1.bincenters[1]
        prof.yerr = h1.yerr
        prof.z    = ratio
        prof.zerr = errors

        prof.labels[0] = h1.labels[0]
        prof.labels[1] = h1.labels[1]
        prof.labels[2] = ylabel
        return prof
    else:
        raise ValueError("can only operator on 1 and 2 dimensional histograms")


def histdiff(h1, h2, ylabel="(h1-h2)/h1", mask_infs=True, relative_error=True):
    if not h1.is_compatible(h2):
        raise ValueError("histograms are not compatible")


    if relative_error:
        ratioerror = lambda x1,x2,dx1,dx2 : n.sqrt( (x2*dx1/n.power(x1,2))**2 + (dx2/x1)**2 )
        diffs = 1 - h2.bincontent/h1.bincontent
        errors = ratioerror( h1.bincontent, h2.bincontent, h1.binerror, h2.binerror)
    else:
        diffs = h1.bincontent - h2.bincontent
        errors = n.sqrt(h1.squaredweights + h2.squaredweights)

    mask = n.ones(shape=h1.bincontent.shape, dtype=bool)
    if mask_infs:
        mask &= n.isfinite(diffs)
        mask &= n.isfinite(errors)

    prof = None
    if h1.ndim == 1:
        prof = scatterpoints.points2d(len(mask[mask]))
        prof.x[:]    = h1.bincenters[mask]
        prof.xerr[:] = h1.binwidths[mask] / 2.
        prof.y[:]    = diffs[mask]
        prof.yerr[:] = errors[mask]
        prof.labels[0] = h1.labels[0]
        prof.labels[1] = ylabel
        return prof
    elif h1.ndim == 2:
        prof = scatterpoints.grid2d( (len(h1.bincenters[0]), len(h1.bincenters[1])) )
        imask = n.logical_not(mask)
        ratio[imask] = n.nan
        errors[imask] = n.nan
        prof.x    = h1.bincenters[0] 
        prof.xerr = h1.xerr
        prof.y    = h1.bincenters[1]
        prof.yerr = h1.yerr
        prof.z    = diffs
        prof.zerr = errors

        prof.labels[0] = h1.labels[0]
        prof.labels[1] = h1.labels[1]
        prof.labels[2] = ylabel
        return prof
    else:
        raise ValueError("can only operator on 1 and 2 dimensional histograms")

def kstest(h1a,h1b, skipemptybins=False):
    """
        perform a Kolmogorov-Smirnov-test on h1a and h1b.
        supported combinations are:
         * h1a & h1b are both hist1d
         * h1a is a hist1d and h1b is a function calculating the exact cdf
         * h1a is a hist1d and h1b is a ndarray containing the exact cdf at 
           the bincenters of h1a

        returns the max. abs. distance of the cdfs and the corresponding
        p-value.
    """
    import scipy.stats
    
    
    if isinstance(h1a, d.histogram.hist1d) and isinstance(h1b, d.histogram.hist1d):
        if not h1a.is_compatible(h1b):
            raise ValueError("histograms must be compatible")

        m = None
        if skipemptybins:
            m = h1a.bincontent > 0
            m &= h1b.bincontent > 0
        else:
            m = n.ones(len(h1a.bincontent), dtype=bool)

        ksdist = n.abs(h1a.bincontent[m].cumsum()/h1a.bincontent[m].sum() - 
                       h1b.bincontent[m].cumsum()/h1b.bincontent[m].sum()).max()
        esum1 = h1a.bincontent[m].sum()**2 / n.power(h1a.binerror[m],2).sum() 
        esum2 = h1b.bincontent[m].sum()**2 / n.power(h1b.binerror[m],2).sum() 
            
        z = ksdist * n.sqrt(esum1*esum2 / (esum1+esum2))
        return ksdist, scipy.stats.ksprob(z)
    elif ((isinstance(h1a, d.histogram.hist1d) and isinstance(h1b, collections.Callable)) or 
          (isinstance(h1a, d.histogram.hist1d) and isinstance(h1b, n.ndarray))):
        
        m = None
        if skipemptybins:
            m = h1a.bincontent > 0
        else:
            m = n.ones(len(h1a.bincontent), dtype=bool)
        
        if isinstance(h1b, collections.Callable):
            h1b = h1b(h1a.bincenters[m])

        if len(h1b) != len(h1a.bincontent[m]):
            raise ValueError("h1a and h1b have unequal lengths")

        esum1 = h1a.bincontent[m].sum()**2 / n.power(h1a.binerror[m],2).sum()
        ksdist = n.abs(h1a.bincontent[m].cumsum()/h1a.bincontent[m].sum() - h1b).max()
        z = ksdist * n.sqrt(esum1)
        return ksdist, scipy.stats.ksprob(z)
    else:
        ValueError("unsupported combination of h1a and h1b. check docs")

def weighted_histsum(histos, ylabel="weighted histsum"):
    for h in histos[1:]:
        if not histos[0].is_compatible(h):
            raise ValueError("histograms have to be compatible")

    if histos[0].ndim != 1:
        raise ValueError("currently only a 1d implementation exists")

    nbins = len(histos[0]._h_bincontent)
    bincontents = n.zeros((len(histos), nbins), dtype=float)
    weights = n.zeros((len(histos), nbins), dtype=float)

    for i,h in enumerate(histos):
        bincontents[i,:] = h._h_bincontent
        #weights[i,:] = 1. / n.power(h.binerror,2)
        weights[i,:] = 1. / h._h_squaredweights

    avg_bincontent = n.zeros(nbins, dtype=float)
    avg_binerror   = n.zeros(nbins, dtype=float)

#    result = scatterpoints.points2d(nbins)
#    result.x[:]    = histos[0].bincenters
#    result.xerr[:] = histos[0].binwidths / 2.

#    for i in xrange(nbins):
#        m = n.isfinite(weights[:,i]) # select histograms with nonempty bins
#        if len(m[m]) > 0:
#            sum1 = (bincontents[:,i][m] * weights[:,i][m]).sum()
#            sum2 = weights[:,i][m].sum()
#            
#            result.y[i] = sum1 / sum2
#            result.yerr[i] = 1. / n.sqrt(sum2)

#        
#    result.labels[0] = histos[0].labels[0]
#    result.labels[1] = ylabel
#    return result

    result =  histos[0].empty_like()
    for i in range(nbins):
        m = n.isfinite(weights[:,i]) # select histograms with nonempty bins
        if len(m[m]) > 0:
            sum1 = (bincontents[:,i][m] * weights[:,i][m]).sum()
            sum2 = weights[:,i][m].sum()
            
            result._h_bincontent[i] = sum1 / sum2
            result._h_squaredweights[i] = 1. / sum2

    return result

