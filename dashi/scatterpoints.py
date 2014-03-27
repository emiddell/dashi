
import numpy as n

class points2d(object):
    def __init__(self, npoints=None):
        if npoints is None:
            self.x = None
            self.y = None
            self.xerr = None
            self.yerr = None
        else:
            self.x = n.zeros(npoints, dtype=float)
            self.y = n.zeros(npoints, dtype=float)
            self.xerr = n.zeros(npoints, dtype=float)
            self.yerr = n.zeros(npoints, dtype=float)
        
        self.labels = [None, None]

class grid2d(object):
    def __init__(self, shape=None):
        if shape is None:
            self.x = None
            self.y = None
            self.z = None
            self.xerr = None
            self.yerr = None
            self.zerr = None
        else:
            self.x = n.zeros(shape, dtype=float)
            self.y = n.zeros(shape, dtype=float)
            self.z = n.zeros(shape, dtype=float)
            self.xerr = n.zeros(shape, dtype=float)
            self.yerr = n.zeros(shape, dtype=float)
            self.zerr = n.zeros(shape, dtype=float)
        
        self.labels = [None, None, None]
