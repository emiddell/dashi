import numpy as np
from scipy import interp,comb
from collections import defaultdict
import sys

################################################################################

#FIXME reference
#limiting cramer-van-mises distribution


cmlimit_t2 = np.asarray( [ .02480, .02878, .03177, .03430, .03656,
                           .03865, .04061, .04247, .04427, .04601,
                           .04772, .04939, .05103, .05265, .05426,
                           .05586, .05746, .05904, .06063, .06222,
                           .06381, .06541, .06702, .06863, .07025,
                           .07189, .07354, .07521, .07690, .07860,
                           .08032, .08206, .08383, .08562, .08744,
                           .08928, .09115, .09306, .09499, .09696,
                           .09896, .10100, .10308, .10520, .10736,
                           .10956, .11182, .11412, .11647, .11888,
                           .12134, .12387, .12646, .12911, .13183,
                           .13463, .13751, .14046, .14350, .14663,
                           .14986, .15319, .15663, .16018, .16385,
                           .16765, .17159, .17568, .17992, .18433,
                           .18892, .19371, .19870, .20392, .20939,
                           .21512, .22114, .22748, .23417, .24124,
                           .24874, .25670, .26520, .27429, .28406,
                           .29460, .30603, .31849, .33217, .34730,
                           .36421, .38331, .40520, .43077, .46136,
                           .49929, .54885, .61981, .74346, 1.16786] )
cmlimit_cumprob = np.hstack( (np.arange(.01,1,.01), [0.999]))

def cm_pvalue(t2):
    if t2 < cmlimit_t2[0]:
        return 0.
    elif t2 > cmlimit_t2[-1]:
        return 1.
    else:
        idx = np.searchsorted(cmlimit_t2, t2)
        return cmlimit_cumprob[idx]


################################################################################

def step_func(step_positions, step_heights):
    """ 
        step function, with steps at step_positions
        step_positions must be sorted
    """
    totalheight = step_heights.sum()
    heights = step_heights.cumsum()

    def stepfunc(x):
        if x < step_positions[0]:
            return 0.
        elif x >= step_positions[-1]:
            return totalheight
        else:
            for i,(bin_min,bin_max) in enumerate(zip(step_positions[:-1], step_positions[1:])):
                if (bin_min <= x) and (x<bin_max):
                    return heights[i]
    return np.vectorize(stepfunc)

################################################################################

def cumdist(where, data, weights):
    """
        calculates the cumulative distribution of data
    """
    data = data.copy()
    idx = data.argsort()
    data = data[idx]
    weights = weights[idx]
    wcum = weights.cumsum() / weights.sum()

    result = np.zeros(len(where), dtype=float)

    for i,x in enumerate(where):
        if x < data[0]:
            result[i] = 0.
        elif x > data[-1]:
            result[i] = 1.
        else:
            result[i] = wcum[np.searchsorted(data, x)]

    return result

def cumdist2(where, data,weights=None):
    """
        where must be sorted
    """
    data = data.copy()
    idx = data.argsort()
    data = data[idx]
    wcum = None
    if weights is None:
        wcum = np.arange(1,len(data)+1,dtype=float) / len(data)
    else:
        weights = weights[idx]
        wcum = weights.cumsum() / weights.sum()

    result = np.zeros(len(where), dtype=float)
    
    j = 0
    for i in xrange(len(where)):
        while (j < len(data)) and (data[j] <= where[i]):
            j += 1

        if j == len(data):
            result[i] = 1.
        elif where[i] < data[0]:
            result[i] = 0
        else:
            result[i] = wcum[j-1]

    return result
    
def cumdist3(where,data,weights):
    data = data.copy()
    idx = data.argsort()
    data = data[idx]
    weights = weights[idx]

    return np.asarray( [weights[data<=i].sum()  for i in where] ) / weights.sum()

def joint_cumdist(data1, data2, weights1=None, weights2=None):
    """
        calculates the cumulative distribution of data
    """
    x = data1.copy()
    y = data2.copy()
    wxcum = None
    wycum = None
    
    if weights1 is None:
        x.sort()
        wxcum = n.arange(1,len(x)+1, dtype=float) / len(x)
    else:
        idx = x.argsort()
        x = x[idx]
        wxcum = weights1[idx].cumsum()
    if weights2 is None:
        y.sort()
        wycum = n.arange(1,len(y)+1, dtype=float) / len(y)
    else:
        idx = y.argsort()
        y = y[idx]
        wycum = weights1[idx].cumsum()
    
    z = np.hstack((x,y))
    zidx = np.arange(1,len(x)+len(y)+1)
    sort_idx = np.argsort(z)
    z = z[sort_idx]
    m = sort_idx < len(x)
    r = zidx[m]
    s = zidx[-m]
    
    i = np.arange(1,len(x)+1)
    j = np.arange(1,len(y)+1)


    #FIXME
    Fslow = np.asarray( [weights1[idx][x<=tmp].sum()  for tmp in x] ) / weights1.sum()
    Gslow = np.asarray( [weights2[idx][y<=tmp].sum()  for tmp in y] ) / weights2.sum()
    Ffast = (wxcum[i-1] + wxcum[s-j-1]) / (weights1.sum() + weights2.sum())
    Gfast = wycum[r-i-1] + wycum[j-1] / (weights1.sum() + weights2.sum() )

    return z,Fslow, Gslow, Ffast, Gfast


################################################################################

def gcd(a,b):
    """ 
        greatest common divisor of a and b calculated 
        (Euclidian algorithm)
    """
    if b == 0:
        return a
    else:
        return gcd(b, a % b)

################################################################################

def lcm(a,b):
    """
        least common multiplier
    """
    return (abs(b) / gcd(a,b) ) * abs(a)

################################################################################


def cmstat_1(data1, data2):
    """
        Cramer-von Mises test statistic
    """

    x = np.asarray(data1, dtype=float)
    y = np.asarray(data2, dtype=float)
    x.sort()
    y.sort()
    lx = float(len(x))
    ly = float(len(y))

    z = np.hstack( (x,y) )
    z.sort()

    #F = cumdist(data1, weights1)
    #G = cumdist(data2, weights2)

    # FIXME: inefficient, no weights
    F = np.asarray( [len(x[x<=i]) / lx for i in z] )
    G = np.asarray( [len(y[y<=i]) / ly for i in z] )

    L =float(lcm(len(x), len(y)))
    h2 = L * (F - G)
    T2 = lx*ly / (lx+ly)**2 * np.power(F - G, 2).sum()
    return T2

def cmstat_1w(data1, data2, weights1, weights2):
    """
        Cramer-von Mises test statistic
    """

    xidx = data1.argsort()
    yidx = data2.argsort()
    x = data1[xidx]
    y = data2[yidx]
    w1 = weights1[xidx]
    w2 = weights2[yidx]
    lx = float(len(x))
    ly = float(len(y))

    z = np.hstack( (x,y) )
    z.sort()

    # FIXME: inefficient
    F = np.asarray( [w1[x<=i].sum()  for i in z] ) / w1.sum()
    G = np.asarray( [w2[y<=i].sum()  for i in z] ) / w2.sum()

    L =float(lcm(len(x), len(y)))
    h2 = L * (F - G)
    T2 = lx*ly / (lx+ly)**2 * np.power(F - G, 2).sum()
    return T2

def cmstat_2(data1,data2):
    x = data1.copy()
    y = data2.copy()
    x.sort()
    y.sort()
    zidx = np.arange(1,len(x)+len(y)+1)
    sort_idx = np.argsort(np.hstack((x,y)))
    m = sort_idx < len(x)
    
    r = zidx[m]
    s = zidx[-m]
    i = np.arange(1,len(x)+1)
    j = np.arange(1,len(y)+1)

    # testing only
    #F = np.vectorize(lambda i : len(x[x<=i]) / float(len(x)))
    #G = np.vectorize(lambda i : len(y[y<=i]) / float(len(y)))
    #assert ( (i/float(len(x)) - (r-i)/float(len(y))) == (F(x)-G(x)) ).all()
    #assert ( ((s-j)/float(len(x)) - (j)/float(len(y))) == (F(y)-G(y)) ).all()

    U = len(x) * np.power((r-i),2).sum() + len(y)*np.power((s-j),2).sum()
    T = float(U) / (len(x)*len(y)*(len(x)+len(y))) - float(4 * len(x)*len(y) - 1) / (6 * (len(x)+len(y)))

    return T,U

def cmstat_3(data1, data2, weights1, weights2):
    x = data1.copy()
    y = data2.copy()
    wxcum = None
    wycum = None
    
    if weights1 is None:
        x.sort()
        wxcum = n.arange(1,len(x)+1, dtype=float) / len(x)
    else:
        idx = x.argsort()
        x = x[idx]
        wxcum = weights1[idx].cumsum()
    if weights2 is None:
        y.sort()
        wycum = n.arange(1,len(y)+1, dtype=float) / len(y)
    else:
        idx = y.argsort()
        y = y[idx]
        wycum = weights1[idx].cumsum()

    zidx = np.arange(1,len(x)+len(y)+1)
    sort_idx = np.argsort(np.hstack((x,y)))
    m = sort_idx < len(x)
    r = zidx[m]
    s = zidx[-m]
    
    i = np.arange(1,len(x)+1)
    j = np.arange(1,len(y)+1)
    z = np.hstack((x,y))
    z.sort()
    FGx = wxcum[i-1] - wycum[r-i-1]
    FGy = wxcum[s-j-1] - wycum[j-1]

    T2 = len(x)*len(y) / float(len(x)+len(y))**2 * (np.power(FGx, 2).sum() + np.power(FGy,2).sum())
    return T2


def cmstat_2sample(x,y,w1=None,w2=None):
    z = np.hstack((x,y))
    z.sort()
    F = cumdist2(z,x,w1)
    G = cumdist2(z,y,w2)
    w1sum = 0
    w2sum = 0
    if w1 is None:
        w1sum = float(len(x))
    else:
        w1sum = w1.sum()

    if w2 is None:
        w2sum = float(len(y))
    else:
        w2sum = w2.sum()
    
    H = (w1sum*F + w2sum*G) / (w1sum + w2sum)
    Hsteps = np.hstack((H[0],H[1:]-H[:-1]))
    T = (w1sum*w2sum) / (w1sum + w2sum) * (np.power(F-G, 2) * Hsteps).sum()
    return T

def cmstat_1sample(cdf,x,weights=None):
    z = x.copy()
    z.sort()
    
    w1sum = 0.
    if weights is None:
        w1sum = float(len(x))
    else:
        w1sum = weights.sum()
    w2sum = float(len(z))

    F = cumdist2(z,x,weights)
    G = cdf(z)
    
    H = (w1sum*F + w2sum*G) / (w1sum + w2sum)
    Hsteps = np.hstack((H[0],H[1:]-H[:-1]))
    T = (w1sum*w2sum) / (w1sum + w2sum) * (np.power(F-G, 2) * Hsteps).sum()
    return T
    


################################################################################

class freq_func(defaultdict):
    """
        how often does a given value occur
    """
    def __init__(self):
        defaultdict.__init__(self, int)

    def __add__(self, other):
        tmp = freq_func()
        for value, freq in self.iteritems():
            tmp[value] += freq
        for value, freq in other.iteritems():
            tmp[value] += freq
        return tmp

    def shift(self, amount):
        tmp = freq_func()
        for value,freq in self.iteritems():
            if freq > 0:
                tmp[value+amount] = freq

        return tmp

    @property
    def values(self):
        "all non-zero values of this frequency function"
        zeta = np.asarray([i for i in self.keys() if self[i] != 0], dtype=float)
        zeta.sort()
        return zeta

    @property
    def frequencies(self):
        "frequencies corresponding to self.values"
        return np.asarray([self[i] for i in self.values])

    def pmf(self, m,n):
        perm = comb(m+n, m)
        T2 = self.values * m * n / ((m+n)**2 * lcm(m,n)**2) # zeta -> T2
        pmf = self.frequencies / perm
        return T2, pmf

    def tail_func(self, i):
        m = self.values >= i
        return self.frequencies[m].sum()


################################################################################
    
def sum_shifted(ff1, ff2, shift):
    tmp = freq_func()
    for value in zip(ff1.keys(), ff2.keys()):
        tmp[value+shift] = ff1[value] + ff2[value]
    return tmp

################################################################################

def identity(z):
    tmp = freq_func()
    tmp[z] = 1
    return tmp

################################################################################


def cm_dist_algo1(m,n):
    L = lcm(m,n)
    a = L / m
    b = L / n

    if m != n: # Algorithm 1
        g = dict()
        for v in xrange(m+1):
            g[(0,v)] = identity(a**2*v*(v+1)*(2*v+1)/6)
        for u in xrange(1,n+1):
            g[(u,0)] = identity(b**2*u*(u+1)*(2*u+1)/6) 
            print "u: %d/%d" % (u,n)
            for v in xrange(1,m+1):
                g[(u,v)] = (g[(u-1,v)] + g[(u,v-1)]).shift( (a*v - b*u)**2 )
            for v in xrange(0,m+1):
                del g[(u-1,v)]
            sys.stdout.flush()

        xs, ys = g[(n,m)].pmf(m,n)

        return g[(n,m)], xs, ys
    if m == n: # Algorithm 1*
        f = dict()
        f[(0,0)] = identity(0)
        for i,t in enumerate(xrange(1,2*n+1)):
            x = min(t, 2*n-t)
            print "%d/%d" % (i, 2*n)
            for d in xrange(x,-1,-2):
                if d==t:
                    f[(t,d)] = identity( t*(t+1)*(2*t+1) / 6)
                else:
                    f[(t,d)] = f[(t-1,d+1)] + f[(t-1,abs(d-1))]
            for d in xrange(x,-1,-2):
                if (t-1,d+1) in f:
                    del f[(t-1,d+1)]
                if (t-1,abs(d-1)) in f:
                    del f[(t-1,abs(d-1))]
        xs, ys = f[(m+n,0)].pmf(m,n)
        return f[(m+n,0)], xs, ys

################################################################################

def tail_convolution(f,g,c):
    " see formula 31 "
    u = f.values 
    v = g.values
    k = len(u)
    l = len(v)
    freq_f = f.frequencies
    freq_g = g.frequencies

    i = 1
    j = k+1
    r = []
    for i in xrange(1, k+1):
        while v[j-1] >= c-u[i-1] and j > 0:
            j -= 1
        r.append(j+1)
    assert len(r) == k

    G = []
    G[0] = freq_g[r[0]-1 : k+1].sum()
    R = freq_f[0] * G[0]
    for i in xrange(1,k+1):
        R += freq_f[i] * ( G[i-1] + freq_g[r[i]:r[i-1]].sum() )

    return R

#def cm_pvalue(m,n,Q):
#    L = lcm(m,n)
#    a = L / m
#    b = L / n
#    M = int( (m+n) / 2 )

#    if m != n: # Algorithm 2
#        g = dict()
#        for v in xrange(m+1):
#            g[(0,v)] = identity(a**2*v*(v+1)*(2*v+1)/6)
#        for u in xrange(1,M+1):
#            g[(u,0)] = identity(b**2*u*(u+1)*(2*u+1)/6) 
#            print "u: %d/%d" % (u,n)
#            for v in xrange(1,min(m,M-u)+1):
#                g[(u,v)] = (g[(u-1,v)] + g[(u,v-1)]).shift( (a*v - b*u)**2 )
#            for v in xrange(1,min(m,M-u)+1):
#                del g[(u-1,v)]
#            sys.stdout.flush()

#        #return g.keys()
#        #xs, ys = g[(n,m)].pmf(m,n)
#        #return g[(n,m)], xs, ys

#    if m == n: # Algorithm 2*
#        f = dict()
#        f[(0,0)] = identity(0)
#        for i,t in enumerate(xrange(1,n+1)):
#            x = min(t, 2*n-t)
#            print "%d/%d" % (i, 2*n)
#            for d in xrange(x,-1,-2):
#                if d==t:
#                    f[(t,d)] = identity( t*(t+1)*(2*t+1) / 6)
#                else:
#                    f[(t,d)] = f[(t-1,d+1)] + f[(t-1,abs(d-1))]
#            for d in xrange(x,-1,-2):
#                if (t-1,d+1) in f:
#                    del f[(t-1,d+1)]
#                if (t-1,abs(d-1)) in f:
#                    del f[(t-1,abs(d-1))]
#        #xs, ys = f[(m+n,0)].pmf(m,n)
#        #return f[(m+n,0)], xs, ys

#    if ( (m+n) % 2 ) == 0:
#        s = 0
#        for v in xrange(0, m+1):
#            w = m-v
#            dv = -b*M + (b+a)*v
#            dw = -b*M + (b+a)*w
#            assert - dv  == dw
#       
#            q = None
#            if m == n:
#                q = tail_convolution(f[(M,dv)], f[(M,dw)], d**2) # FIXME who the fuck is d**2?

