"""
    The histfactory module provides functions to easily create
    one or two dimensional histograms, estimating the necessary
    binning and fill them.
"""
import dashi.histogram as histogram
import dashi.histfuncs as histfuncs
import numpy as n


def hist1d(sample, bins, weights=None, label=None, title=None):
    """
        factory method to create one dimensional histograms

        **Parameters:**
            *sample* : [ numpy.ndarray | list ]
               values that will be filled into the histogram
            *bins* : [ int | tuple | numpy.ndarray ]
                can be either the number of bins, a tuple (minvalue, maxvalue,
                nbins), or a numpy array denoting the bin edges.  If a number
                is provided the value range of the sample is determined,
                enlarged by a small tolerance and divided into nbins.  If a
                tuple is provided the given range is divided into nbins. Two
                extra bins are added left and right.  If the bin edges are
                specified they are used unmodified.
            *weights* : [None | numpy.ndarray | list]
                If provided each sample[i] will add weights[i] to its bin. Otherwise
                each sample[i] contributes 1.
            *label* : [None, str]
                a string labeling the variable that is filled into the histogram
                (in the current implementation this will be used by the
                plotting routines to set the x-label)
            *title* : [None,str]
                a string labeling the histogram

        **Return Value:**
            the filled :class:`dashi.histogram.hist1d`
                
    """
    
    if not isinstance(sample, n.ndarray):
        sample = n.asarray(sample)

    if (not weights is None) and (not isinstance(weights, n.ndarray)):
        weights = n.asarray(weights)
    
    if not isinstance(bins, n.ndarray):
        if isinstance(bins, tuple):
            bins = histfuncs.generatebins_1d_tuple(bins)
        else:
            bins = histfuncs.generatebins_1d(sample, bins)


    h = histogram.hist1d(bins, label=label, title=title)
    h.fill(sample, weights)
    return h


def hist2d(sample, bins, weights=None, labels=None, title=None):
    """
        factory method to create two dimensional histograms

        **Parameters:**
            *sample* : tuple
                a tuple of two arrays or lists of eqal size. The values sample[0][i] and
                sample[1][i] will be treated as one event.
            *bins* : [int | tuple]
                can be either 
                    * one number yielding the same number of bins in both dimensions
                    * a tuple with two numbers,  denoting the number of bins for each dimension
                    * a tuple of numpy.ndarrays giving the binedges
            *weights* : [None | list | numpy.ndarray]
                If provided the event given by sample[0][i] and sample[1][i]
                will be filled into the histogram as one entry with weight
                weights[i]. Otherwise it contributes 1 to its bin.
            *labels* : [None, tuple]
                A tuple of strings labeling the variables that are filled into the histogram.
                (in the current implementation these will be used by the
                plotting routines to set the x- and y-labels)
            *title* : [None, str]
                a string labeling the histogram
        
        **Return Value:**
            the filled :class:`dashi.histogram.hist2d`
    """
    if not (isinstance(bins, tuple) and all(map(lambda i: isinstance(i, n.ndarray), bins))):
        bins = histfuncs.generatebins_nd(2, sample, bins)
    
    if isinstance(sample, tuple) and (all(map(lambda i: isinstance(i, list), sample))):
        sample = tuple( n.asarray(i) for i in sample )

    if not weights is None:
        if isinstance(weights, tuple) and (all(map(lambda i: isinstance(i, list), weights))):
            weights = tuple( n.asarray(i) for i in weights )
        

    h = histogram.hist2d(bins, labels=labels, title=title)
    h.fill(sample, weights)
    return h
