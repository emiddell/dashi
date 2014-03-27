#
# stuff that fits nowhere else
#

import numpy as n

def calc_ylim(histlist, log, top=.2, bottom=0.):
    """ takes a list of histograms and calculcates reasonable min and max values 
        that can be given to pylab.ylim
        
        Parameters:
          log   : the calculation is slightly different for linear and logarithmic y-axes
          top   : fraction of space to reserve at the top of the plot
          bottom: fraction of space to reserve at the bottom of the plot
    """

    # loop through all histograms and find min and max values
    maximums = n.zeros(len(histlist), dtype=float)
    minimums = n.zeros(len(histlist), dtype=float)
    minimums_nozero = n.zeros(len(histlist), dtype=float)

    for i,hist in enumerate(histlist):
        maximums[i] = hist.bincontent.max()
        minimums[i] = hist.bincontent.min() 
        if not (hist.bincontent == 0).all():
            minimums_nozero[i] = hist.bincontent[hist.bincontent!=0].min()
        else:
            minimums_nozero[i] = n.nan

    # select only those histograms where min&max values are finite
    m = n.isfinite(minimums) & n.isfinite(maximums) & n.isfinite(minimums_nozero)

    # if no histogram provided useful information just return (0,1)
    if len(minimums_nozero[m]) == 0:
        return(0,1)
    else:
        if log:
            mi = minimums_nozero[m].min()
            ma = maximums[m].max()
            range=(n.log10(ma)-n.log10(mi)) / (1. - float(top) - float(bottom))
            ma = ma * 10**(range*top)     # ma = 10**(n.log10(ma) + range*top)
            mi = mi * 10**(-range*bottom) # mi = 10**(n.log10(mi) - range*bottom)
            return (mi,ma)
        else:
            mi = minimums.min()
            ma = maximums.max()
            range=(ma-mi) / (1. - float(top) - float(bottom))
            ma += range*top
            mi -= range*bottom
            return(mi,ma)
