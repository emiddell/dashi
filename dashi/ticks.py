
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms

import math
import numpy as np
 

def coverage(dmin, dmax, lmin, lmax):
    return 1. - 0.5 * ( (dmax-lmax)**2 + (dmin - lmin)**2 ) / (0.1 * (dmax - dmin) )**2

def coverage_max(dmin, dmax, span):
    drange = dmax - dmin
    if span > drange:
        return 1. - (0.5*(span-drange))**2 / (0.1 * drange)**2
    else:
        return 1.

def density(k,m,dmin,dmax,lmin,lmax):
    r  = (k - 1.) / (lmax -lmin)
    rt = (m - 1.) / (max(lmax,dmax) - min(lmin,dmin))
    return 2. - max(r/rt, rt / r)

def density_max(k,m):
    if k >= m:
        return 2. - (k - 1.0) / (m - 1.0)
    else:
        return 1.
    

def simplicity(q, Q, j, lmin, lmax, lstep):
    eps = 1e-10
    n = len(Q)
    i = Q.index(q) + 1
    v = 0
    if ((lmin % lstep) < eps) or ( (((lstep-lmin) % lstep) < eps) and (lmin <= 0) and (lmax >= 0)):
        v = 1
    else:
        v = 0

    return (n - i) / (n - 1.0) + v - j

def simplicity_max(q,Q,j):
    n = len(Q)
    i = Q.index(q) + 1
    v = 1
    return (n - i) / (n - 1.0) + v - j

def legibility(lmin, lmax, lstep):
    return 1.

def score(weights, simplicity, coverage,density, legibility):
   return weights[0]*simplicity + weights[1]*coverage + weights[2]*density + weights[3]*legibility  

def wilk_ext(dmin, dmax, m, only_inside, 
             Q=[1, 5, 2, 2.5, 4, 3], 
             w=[0.2, 0.25, 0.5, 0.05]):

    if (dmin >= dmax) or (m < 1):
        return (dmin, dmax, dmax - dmin, 1, 0, 2, 0)
        
    n = float(len(Q))
    best_score = -1.0
    result = None

    j = 1.0
    while (j < np.inf):
        for q in map(float, Q):
            sm = simplicity_max(q,Q,j)

            if score(w,sm, 1,1,1) < best_score:
                j = np.inf
                break

            k = 2.
            while k < np.inf:
                dm = density_max(k,m) 

                if score(w,sm,1,dm,1) < best_score:
                    break

                delta = (dmax - dmin) / (k + 1.) / j / q
                z = math.ceil(math.log10(delta))

                while z < np.inf:
                    step = j * q * 10.**z
                    cm = coverage_max(dmin, dmax, step * (k-1.))
                    
                    if score(w,sm,cm,dm,1) < best_score:
                        break

                    min_start = math.floor(dmax/step)* j - (k-1.) * j
                    max_start = math.ceil(dmin/step) * j

                    if min_start > max_start:
                        z += 1
                        break

                    for start in np.arange(min_start,max_start+1):
                        lmin  = start * (step/j)
                        lmax  = lmin + step * (k-1.0)
                        lstep = step

                        s = simplicity(q,Q,j,lmin,lmax,lstep)
                        c = coverage(dmin,dmax,lmin,lmax)
                        d = density(k,m,dmin,dmax,lmin,lmax)
                        l = legibility(lmin,lmax,lstep)
                        scr = score(w,s,c,d,l)

                        if (scr > best_score) and \
                           ((only_inside <= 0) or ((lmin >= dmin) and (lmax <=dmax))) and \
                           ((only_inside >= 0) or ((lmin <= dmin) and (lmax >= dmax))):
                            best_score = scr
                            #print "s: %5.2f c: %5.2f d: %5.2f l: %5.2f" % (s,c,d,l)
                            result =  (lmin,lmax,lstep,j,q,k,scr)

                    z += 1
                # end of z-while-loop    
                k += 1
            # end of k-while-loop
        j += 1
    # end of j-while-loop
    return result 

class ExtendedWilkinsonTickLocator(mticker.Locator):
    """
        places the ticks according to the extended Wilkinson algorithm
        (http://vis.stanford.edu/files/2010-TickLabels-InfoVis.pdf)

        **Parameters:**
            *target_density* : [ float ]
                controls the number of ticks. The algorithm will try
                to put as many ticks on the axis but deviations are 
                allowed if another criterion is considered more important.
                
            *only_inside* : [ int ]
                controls if the first and last label include the data range.
                0  : doesn't matter
                >0 : label range must include data range
                <0 : data range must be larger than label range 

            *Q* : [ list of numbers ]
                numbers that are considered as 'nice'. Ticks will be
                multiples of these
            
            *w* : [ list of numbers ]
                list of weights that control the importance of the four
                criteria: simplicity, coverage, density and legibility
            
                see the reference for details
    """
    def __init__(self, target_density, only_inside=0, 
                 Q=[1, 5, 2, 2.5, 4, 3],
                 w=[0.2, 0.25, 0.5, 0.05]):


        self.target_density = target_density
        self.only_inside = only_inside
        self.Q = Q
        self.w = w

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        if vmax<vmin:
            vmin, vmax = vmax, vmin

        lmin,lmax,lstep,j,q,k,scr = wilk_ext(vmin, vmax, self.target_density, self.only_inside, self.Q, self.w)
        #print "Ticking:" + ((7*"%f ") % (lmin,lmax,lstep,j,q,k,scr))
        return np.arange(lmin,lmax+lstep,lstep)

class PaddedLogLocator(mticker.LogLocator):
    def view_limits(self, vmin, vmax):
        """
        Try to choose the view limits intelligently. In contrast to the stock
        LogLocator, this version always pads by one decade so that the contents
        of the largest and smallest histogram bins can be seen even if they
        fall on a decade.
        """
        b = self._base

        if vmax < vmin:
            vmin, vmax = vmax, vmin

        if self.axis.axes.name == 'polar':
            vmax = math.ceil(math.log(vmax) / math.log(b))
            vmin = b ** (vmax - self.numdecs)
            return vmin, vmax

        minpos = self.axis.get_minpos()

        if minpos <= 0 or not np.isfinite(minpos):
            raise ValueError(
                "Data has no positive values, and therefore can not be "
                "log-scaled.")

        if vmin <= minpos:
            vmin = minpos
        
        vmin = mticker.decade_down(vmin*(1-1e-10), self._base)
        vmax = mticker.decade_up(vmax*(1+1e-10), self._base)

        if vmin == vmax:
            vmin = mticker.decade_down(vmin, self._base)
            vmax = mticker.decade_up(vmax, self._base)
        result = mtransforms.nonsingular(vmin, vmax)
        return result