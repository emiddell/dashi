"""
 tool to interactively inspect the influence of cuts
 based on an idea of Robert Franke
"""

import re
import numpy as n
import pylab as p
from matplotlib.widgets import Slider, Button

import dashi as d


class VisualCutter(object):
    def __init__(self, varbundle, weights, cutstring, initfunc, updatefunc, basemask=None):
        """
            varbundle: varnames -> categories -> numpy arrays
        """
        self.vars = varbundle
        self.weights = weights

        self.varnames = self.vars.keys()
        self.catnames = self.vars.transpose().keys()

        self.initfunc = initfunc
        self.updatefunc = updatefunc

        self.cutstring = cutstring

        if basemask is None:
            self.basemask = d.bundle(**dict([(k, n.ones(len(weights.get(k)), dtype=bool)) for k in weights.keys()]))
        else:
            self.basemask = basemask

        self.ranges = dict()

        placeholders = re.findall("\%\((\w+)\)\w", cutstring)
        for ph in placeholders:
            if ph not in self.varnames:
                print "Couldn't identify key %s. Set range in self.ranges manually!" % ph
                self.ranges[ph] = None
            else:
                mi = n.nanmin( self.vars.get(ph).map(n.nanmin).values())
                ma = n.nanmax( self.vars.get(ph).map(n.nanmax).values())
                self.ranges[ph] = (mi,ma)



    def run(self):
        for varname,range in self.ranges.iteritems():
            if range is None:
                raise ValueError("no range specified for slider %s." % varname)

        def update(val):
            d = dict([(k, self.sliders[k].val) for k in self.sliders.keys()])
            # pull members into current scope -> no self. in cutstring
            vars = self.vars
            weights = self.weights
            basemask = self.basemask

            # evaluate cutstring and combine with basemask
            mask = basemask & eval(self.cutstring % d)
            self.updatefunc(self, vars, weights, mask)

        self.sliderfig = p.figure(figsize=(3,len(self.ranges)*.4))
        self.sliders = dict()
        space = .05
        height = (1. - (len(self.ranges)+2)*space) / float(len(self.ranges))
        
        for i,(varname,range) in enumerate(self.ranges.iteritems()):
            ax = p.axes([0.25, 1-float(i+1)*(space+height), 0.5, height])
            slider = Slider(ax, varname, range[0],range[1], valinit=(range[1]-range[0])/2.)
            slider.on_changed(update)
            self.sliders[varname] = slider
        
        self.initfunc(self,self.vars,self.weights,self.basemask)





def test():
    sig = d.bundle(var1 = n.random.normal(2,2,1e5), var2 = n.random.normal(1,1,1e5))
    bg  = d.bundle(var1 = n.random.normal(0,1,1e4), var2 = n.random.normal(-1,2,1e4))
    vars = d.bundle(sig=sig, bg=bg).transpose()
    weights = d.bundle(sig=n.ones(1e5), bg=n.ones(1e4))

    hist1d = d.bundleize(d.factory.hist1d)
    d.visual()

    def initfunc(vc,vars,weights,mask):
        vc.myfig = p.figure()
        p.figure(vc.myfig.number)
        h1 = hist1d( vars.var1[mask], n.linspace(-20,20,101), weights[mask])
        c = d.bundle(sig="r", bg="k")
        h1.line(c=c)

    def updatefunc(vc,vars,weights,mask):
        vc.myfig.clear()
        p.figure(vc.myfig.number)
        h1 = hist1d( vars.var1[mask], n.linspace(-20,20,101), weights[mask])
        c = d.bundle(sig="r", bg="k")
        h1.line(c=c)
        vc.myfig.canvas.draw()

    def anyfunc(*args):
        print args

    c = VisualCutter(vars,weights, "(vars.var1 > %(var1)s) & (vars.var2 < %(var2)s)", initfunc, updatefunc)
    c.run()


