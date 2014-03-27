
import pylab as p
import numpy as n
import matplotlib as mpl
from infobox import InfoBox 
from odict import OrderedDict
import dashi.histfuncs
import dashi as d
import copy

def _h1label(h1):
    label = h1.labels[0]
    if label is not None:
        ax = p.gca()
        if ax.get_xlabel() == '':
            ax.set_xlabel(label)

def _h2label(h2, orientation='horizontal'):

    xlabel = h2.labels[0]
    ylabel = h2.labels[1]
    if orientation == 'vertical':
        xlabel, ylabel = ylabel, xlabel
    ax = p.gca()
    if ax.get_xlabel() == '' and ax.get_ylabel() == '':
        if (xlabel is not None):
            ax.set_xlabel(xlabel)
        if (ylabel is not None):
            ax.set_ylabel(ylabel)

class LegendProxy(object):
    """
    Provide proxies to draw legend entries for unsupported artist types.
    """
    def __init__(self, ax):
        self.ax = ax
        # replace Axes.legend() with a call to ourselves
        ax._legend = ax.legend
        ax.legend = self.__call__
        self.artists = []
        self.labels = []
    
    def _validate(self, **kwargs):
        if not 'label' in kwargs:
            return False
        if kwargs['label'] is None or kwargs['label'].startswith('__nolabel'):
            return False
        return True
    
    def add_line(self, **kwargs):
        if not self._validate(**kwargs):
            return
        line = mpl.lines.Line2D([0,0], [1,1], linewidth=kwargs.get('linewidth', 1), color=kwargs.get('color', None), linestyle=kwargs.get('ls', None))
        self.artists.append(line)
        self.labels.append(kwargs.get('label', None))
    
    def add_fill(self, **kwargs):
        if not self._validate(**kwargs):
            return
        rect = mpl.patches.Rectangle((0, 0), 1, 1, color=kwargs.get('color', None), alpha=kwargs.get('alpha', 1.))
        self.artists.append(rect)
        self.labels.append(kwargs.get('label', None))
    
    def add_scatter(self, **kwargs):
        if not self._validate(**kwargs):
            return
        if not 'color' in kwargs:
            colors = None
        else:
            colors = mpl.colors.colorConverter.to_rgba_array(kwargs['color'], 1.0)
        artist = mpl.collections.AsteriskPolygonCollection(4, 0, [20., 20.], facecolors=colors, edgecolors=colors, offsets=[[0,0], [1,1]])
        self.artists.append(artist)
        self.labels.append(kwargs.get('label', None))

    def __call__(self, **kwargs):
        artists, labels = self.ax.get_legend_handles_labels()
        for a, l in zip(self.artists, self.labels):
            if l is not None and not l.startswith('__nolabel'):
                artists.append(a)
                labels.append(l)
        if len(artists) > 0:
            return self.ax._legend(artists, labels, **kwargs)
        else:
            return self.ax._legend(**kwargs)

def _h1_transform_bins(self, differential=False, cumulative=False, cumdir=1):
    """
    Transform bin contents in errors to values appropriate for a differential
    or cumulative histogram.
    
    For cumdir=1, the bins contain the sum of weights associated with values
    <= to the *right-hand* bin edge; for cumdir=-1, the bin value is the sum
    of weights for values > the *left-hand* bin edge.
    
    (JvS)
    """
    if (cumulative and differential):
        raise ValueError("cumulative and differential are mutually exclusive!")
    if cumulative:
        if cumdir == 1:
            vis = slice(1, -1)
            bincontent = n.cumsum(self._h_bincontent)[vis]
            binerror   = n.sqrt(n.cumsum(self._h_squaredweights))[vis]
        elif cumdir == -1:
            vis = slice(0, -2)
            bincontent = self._h_bincontent.sum() - n.cumsum(self._h_bincontent)[vis]
            binerror  = n.sqrt( (self._h_squaredweights).sum() - n.cumsum(self._h_squaredweights)[vis] )
        else:
            raise ValueError("cumdir should be 1 or -1")
    elif differential:
        bincontent = self.bincontent/self.binwidths
        binerror   = self.binerror/self.binwidths
    else:
        bincontent = self.bincontent
        binerror   = self.binerror
    return bincontent, binerror

def h1scatter(self, log=False, cumulative=False, cumdir=1, color=None, differential=False, **kwargs):
    """ use pylab.errorplot to plot a 1d histogram 
        
        Parameters:
          log        : if true create logartihmic plot
          cumulative : plot the cumulative histogram

          (all other kwargs will be passed to pylab.errobar)
    """

    if self.stats.weightsum == 0:
        return
    
    bincontent, binerror = _h1_transform_bins(self, differential, cumulative, cumdir)
    
    ax = p.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    
    kw = {
        "xerr" : self.xerr,
        "yerr" : binerror,
        "fmt" : "k",
        "capsize" : 0.,
        "linestyle" : 'None',
        "color" : color,
    }
    # for log-scaled axes, clip the lower extent of the error bar
    if log:
        kw["yerr"] = [n.where(bincontent-binerror <= 0, abs(bincontent)*(1-1e-12), binerror), binerror]

    patches = None

    kw.update(kwargs)
    if len(ax.lines) > 0:
        minvalue,maxvalue = p.ylim()
    else:
        minvalue,maxvalue = float('inf'),-float('inf')
    
    if not hasattr(ax, "_legend_proxy"):
        ax._legend_proxy = LegendProxy(ax)
    label = kw.pop('label', self.title)
    ax._legend_proxy.add_scatter(label=label, **kw)
    
    if not log:
        if self.stats.weightsum > 0:
            minvalue = min( bincontent.min(), minvalue )
            maxvalue = max( 1.3*bincontent.max(), maxvalue)
            patches = p.errorbar(self.bincenters, bincontent, **kw) 
        else:
            p.xlim(self.binedges[0], self.binedges[-1])
    else:
        ax = p.gca()
        ax.set_yscale("log", nonposy='clip')
        ymi,yma = p.ylim()
        if len(p.gca().get_lines()) != 0:
            ymi = min([i._y[i._y > 0].min() for i in p.gca().get_lines()])
        if ymi == 0:
            ymi = float('inf')
        if self.stats.weightsum > 0:
            lower = bincontent[bincontent > 0]-binerror[bincontent>0]
            nonzerolower = lower[lower>0]
            if len(nonzerolower)>0:
                if nonzerolower.min() > 0:
                    minvalue = min(0.1 * nonzerolower.min(), ymi)
                else:
                    minvalue = min(0.05* lower.min(), ymi)
            else:
                minvalue = ymi
            maxvalue = max( n.power(10, n.ceil (n.log10(100. * bincontent.max()))), maxvalue)
            patches = p.errorbar(self.bincenters, bincontent, **kw) 
        else:
            p.xlim(self.binedges[0], self.binedges[-1])
    
    p.ylim(minvalue, maxvalue)
    _h1label(self)

    return patches

def h1band(self, log=False, type="steps", differential=False, cumulative=False, cumdir=1, **kwargs):
    """ plot a filled band to indicate bincontents and binerrors 
    
        Parameters:
          log : if true create logartihmic plot
          cumulative : plot the cumulative histogram
          type: ['line', 'steps']

          (all other kwargs will be passed to pylab.fill_between)
    """
        
    if self.stats.weightsum == 0:
        return

    bincontent, binerror = _h1_transform_bins(self, differential, cumulative, cumdir)

    if type.lower() == "line":
        x  = self.bincenters
        y1 = bincontent - binerror
        y2 = bincontent + binerror
    elif type.lower() == "steps":
        nbins = self.nbins[0]
        x = n.zeros(2*(nbins+1), dtype=float)
        y1 = n.zeros(2*(nbins+1), dtype=float)
        y2 = n.zeros(2*(nbins+1), dtype=float)

        for i in xrange(nbins):
            x[1+2*i] = self.binedges[i]
            x[2+2*i] = self.binedges[i+1]
            y1[1+2*i] = bincontent[i] - binerror[i]
            y1[2+2*i] = bincontent[i] - binerror[i]
            y2[1+2*i] = bincontent[i] + binerror[i]
            y2[2+2*i] = bincontent[i] + binerror[i]

        # left-most point
        x[0] = self.binedges[0]
        y1[0] = 0
        y2[0] = 0
        # right-most point
        x[-1] = self.binedges[-1]
        y1[-1] = 0
        y2[-1] = 0
    else:
        raise ValueError("did't understand given type %s" % type)

    minvalue,maxvalue = p.ylim()
    if log:
        ax = p.gca()
        ax.set_yscale("log", nonposy='clip')
        ymi,yma = p.ylim()
        nonemptybins = bincontent[bincontent > 0]
        if len(nonemptybins) > 0:
            minvalue = min(0.1 * nonemptybins.min(), ymi)
            maxvalue = max( n.power(10, n.ceil (n.log10(100. * nonemptybins.max()))), maxvalue)
    else:
        minvalue = min( bincontent.min(), minvalue )
        maxvalue = max( 1.3*bincontent.max(), maxvalue)

    ax = p.gca()
    if not hasattr(ax, "_legend_proxy"):
        ax._legend_proxy = LegendProxy(ax)
    label = kwargs.pop('label', self.title)
    ax._legend_proxy.add_fill(label=label, **kwargs)

    kw = {}
    kw.update(kwargs)
    p.fill_between(x, y1, y2, **kw)

    
    p.ylim(minvalue, maxvalue)
    _h1label(self)

def h1line(self, log=False, cumulative=False, differential=False, cumdir=1, filled=False, color=None, orientation='horizontal', **kwargs):
    """ plot the histogram's  bincontent with a line using pylab.plot. 
        Parameters:
          log    : if true create logartihmic plot
          cumulative : plot the cumulative histogram
          filled : if true fill the region below the line 

          (all other kwargs will be passed to pylab.plot or pylab.fill)
          Note: pylab.plot and pylab.fill take quite different kwargs.
    """
    
    bincontent, binerror = _h1_transform_bins(self, differential, cumulative, cumdir)
    
    nonzerobc = bincontent[bincontent > 0]
    if len(nonzerobc) == 0:
        return

    nbins = self.nbins[0]

    xpoints = n.zeros(2*(nbins+1), dtype=float)
    ypoints = n.zeros(2*(nbins+1), dtype=float)

    for i in xrange(nbins):
        xpoints[1+2*i] = self.binedges[i]
        xpoints[2+2*i] = self.binedges[i+1]
        ypoints[1+2*i] = bincontent[i]
        ypoints[2+2*i] = bincontent[i]

    xpoints[0] = self.binedges[0]
    ypoints[0] = 0
    xpoints[-1] = self.binedges[-1]
    ypoints[-1] = 0
    # TODO eventually add another point to close area?
    
    ax = p.gca()
    if orientation == 'vertical':
        xpoints, ypoints = ypoints, xpoints
        ylim = p.xlim
        set_yscale = lambda v: ax.set_xscale(v, nonposx='clip')
    else:
        ylim = p.ylim
        set_yscale = lambda v: ax.set_yscale(v, nonposy='clip')

    minvalue,maxvalue = ylim()
    if minvalue == 0 and maxvalue == 1:
        maxvalue=0.
    if log:
        set_yscale("log")
        if len(nonzerobc) > 0:
            minvalue = min( n.power(10, n.floor(n.log10(0.1 * nonzerobc.min()))) , minvalue)
            maxvalue = max( n.power(10, n.ceil (n.log10(100. * bincontent.max()))), maxvalue)
            ypoints[ypoints == 0] = minvalue
    else:
        set_yscale("linear")
    
    if not hasattr(ax, "_legend_proxy"):
        ax._legend_proxy = LegendProxy(ax)
    label = kwargs.pop('label', self.title)
    if color is None:
        color = ax._get_lines.color_cycle.next()
    kwargs['color'] = color
    if filled:
        ax._legend_proxy.add_fill(label=label, **kwargs)
    else:
        ax._legend_proxy.add_line(label=label, **kwargs)
    
    if filled:
        kw = {"ec":"k", "fc": color}
        kw.update(kwargs)
        p.fill(xpoints, ypoints, **kw) 
        minvalue = min( bincontent.min(), minvalue )
        maxvalue = max( 1.3*bincontent.max() , maxvalue )
    else:
        kw = {"color":color}
        kw.update(kwargs)
        p.plot(xpoints, ypoints, "k-", **kw) 
        minvalue = min( bincontent.min(), minvalue )
        maxvalue = max( 1.3*bincontent.max() , maxvalue )

    ylim(minvalue, maxvalue)
    _h1label(self)

def _h2_transform_bins(self, kwargs):
    """
    Common manipulation for 2-d plots
    """
    bincontent = self.bincontent.T.copy()
    
    cumdir = kwargs.pop("cumdir", None)
    if cumdir is not None:
        bincontent = self._h_bincontent.cumsum(axis=1).cumsum(axis=0)[1:-1,1:-1].T
    
    _process_colorscale(bincontent, kwargs)
    return bincontent, kwargs

def _process_colorscale(bincontent, kwargs):
    if not kwargs.get("norm", None):
        if kwargs.pop("log", False):
            from matplotlib.colors import LogNorm
            minval = 10**n.floor(n.log10(bincontent[bincontent>0].min()))
            maxval = bincontent[bincontent>0].max()
            kwargs["norm"] = LogNorm(vmin=minval, vmax=maxval)
        else:
            from matplotlib.colors import Normalize
            bincontent[bincontent==0] = n.nan
            kwargs["norm"] = Normalize(vmin=0, vmax=n.nanmax(bincontent))
    
    # Make sure the color map has an underflow color
    cmap = kwargs.pop("cmap", p.cm.jet)
    cmap = copy.deepcopy(cmap)
    cmap.set_under(color='w')
    cmap.set_bad(color='w')
    kwargs["cmap"] = cmap
    return kwargs

def h2imshow(self, **kwargs):
    """ plot the 2d histogram's  bincontent with pylab.imshow 
        Parameters:
          log    : if true create logartihmic plot. Empty
                   bins will be filled up the lowest populated decade.

          (all other kwargs will be passed to pylab.imshow)
    """
    bincontent, kwargs = _h2_transform_bins(self, kwargs)

    kw = {"cmap": mpl.cm.jet, "aspect" : "auto", "interpolation" : "nearest" }
    kw.update(kwargs)
    img = p.imshow(bincontent, origin="lower", 
             extent=(self.binedges[0][0], self.binedges[0][-1], self.binedges[1][0], self.binedges[1][-1]), 
             **kw) 
    _h2label(self)
    
    return img

def h2pcolor(self, **kwargs):
    """ plot the 2d histogram's  bincontent with pylab.pcolor 
        Parameters:
          log    : if true create logartihmic plot. Empty
                   bins will be filled up the lowest populated decade.

          (all other kwargs will be passed to pylab.pcolor)
    """
    bincontent, kwargs = _h2_transform_bins(self, kwargs)

    img = p.pcolor(self.binedges[0], self.binedges[1], bincontent, **kwargs)
    _h2label(self)
    
    return img

def h2pcolormesh(self, **kwargs):
    """ plot the 2d histogram's  bincontent with pylab.pcolormesh 
        Parameters:
          log    : if true create logartihmic plot. Empty
                   bins will be filled up the lowest populated decade.

          (all other kwargs will be passed to pylab.pcolor)
    """
    bincontent, kwargs = _h2_transform_bins(self, kwargs)

    img = p.pcolormesh(self.binedges[0], self.binedges[1], bincontent, **kwargs)
    _h2label(self)
    
    return img

def h2contour(self, levels=10, filled=False, clabels=False, **kwargs):
    """ plot the 2d histogram's  bincontent with pylab.imshow 
        Parameters:
          log    : if true create logartihmic plot. Empty
                   bins will be filled up the lowest populated decade.
          filled : toggle beetween pylab.contour and pylab.contourf

          (all other kwargs will be passed to pylab.contour/pylab.contourf)
    """
    bincontent = self.bincontent.T.copy()

    if kwargs.pop("log", False):
        if bincontent.sum() > 0:
            #minval = 10**n.ceil(n.log10(bincontent[bincontent>0].min()))
            bincontent[bincontent==0] = n.nan # won't be drawn by imshow
            bincontent = n.log10(bincontent) # log of nans are nans
    else:
        if bincontent.sum() > 0:
            bincontent[bincontent == 0] = n.nan
        
         

    kw = {}
    kw.update(kwargs)
    plfunc = None 
    cs = None
    if filled:
        cs = p.contourf(self.bincenters[0], self.bincenters[1], bincontent, levels, **kw) 
    else:
        cs = p.contour(self.bincenters[0], self.bincenters[1], bincontent, levels, **kw) 

    if clabels:
        p.clabel(cs, inline=1)

    _h2label(self)
    
    return cs

def might_be_logspaced(bins):
    nslog = n.unique(bins[1:]/bins[:-1])
    nslin = n.unique(n.diff(bins))
    return len(nslog) < len(nslin)

def h2stack1d(self, boxify=False, cmap=mpl.cm.jet, colorbar=True, cumdir=1, **kwargs):
    histos = []
    for i in xrange(self.nbins[1]):
        histos.append(d.factory.hist1d(self.bincenters[0], self.binedges[0], weights=self.bincontent[:,i]))
    
    if cumdir >= 0:
        histocum = [reduce(lambda i,j : i+j, histos[:end]) for end in xrange(1,len(histos)+1)]
        it = reversed(list(enumerate(zip(histocum, self.bincenters[1]))))
    else:
        histocum = [reduce(lambda i,j : i+j, histos[begin:]) for begin in xrange(0,len(histos)-1)]
        it = iter(list(enumerate(zip(histocum, self.bincenters[1]))))
    
    log = might_be_logspaced(self.binedges[1])
    if log:
        trafo = mpl.colors.LogNorm(vmin=self.bincenters[1][0], vmax=self.bincenters[1][-1])
    else:
        trafo = mpl.colors.Normalize(vmin=self.bincenters[1][0], vmax=self.bincenters[1][-1])
    
    for i,(h,cvalue) in it:
        if boxify:
            mask = histocum[-1]._h_bincontent > 0
            hscale = h.empty_like()
            hscale._h_bincontent[mask] = (h._h_bincontent / histocum[-1]._h_bincontent)[mask]
            hscale.line(filled=1, fc=cmap(trafo(cvalue)), **kwargs)
        else:
            h.line(filled=1, fc=cmap(trafo(cvalue)), **kwargs)
    p.xlim(self.binedges[0][0], self.binedges[0][-1])
    if boxify:
        p.ylim(0,1.05)
    
    cbargs = dict(cmap=cmap, norm=trafo, orientation='vertical')
    if log:
        # logspaced colorbars need a little hand-holding
        from matplotlib.ticker import LogFormatter
        cbargs['ticks'] = self.binedges[1]
        cbargs['format'] = LogFormatter()
    
    if colorbar:
        ax = p.gca()
        cax,cax_kw = mpl.colorbar.make_axes(ax)
        cax.hold(True)
        cb1 = mpl.colorbar.ColorbarBase(cax, **cbargs)
        p.gcf().sca(ax)
        if self.labels[0] is not None:
            ax.set_xlabel(self.labels[0])
        if self.labels[1] is not None:
            cb1.set_label(self.labels[1])
    else:
        return cbargs



def h1statbox(self, axes=None, name=None, loc=2, fontsize=10, prec=5, other=None, **kwargs):
    assert self.ndim == 1

    stringdict = OrderedDict()

    formatfunc = dashi.histfuncs.number_format

    if not name:       
        name = None   #empty string causes pylab crash
    
    stringdict["N"]     = formatfunc(self.stats.weightsum, prec)
    stringdict["mean"]  = formatfunc(self.stats.mean, prec)
    stringdict["std"]   = formatfunc(self.stats.std, prec) 
    stringdict["med"]   = formatfunc(self.stats.median, prec)
    stringdict["uflow"] = formatfunc(self.stats.underflow, prec)
    stringdict["oflow"] = formatfunc(self.stats.overflow, prec)
    stringdict["nans"] = formatfunc(self.stats.nans_wgt, prec)

    if other is not None:
        if not isinstance(other, dict):
            raise ValueError("other must be a dict")
        
        for key, func in other.iteritems():
            stringdict[key] = formatfunc(func(self), prec)

    infobox = InfoBox(stringdict,title=name, **kwargs) # need to pop kwargs
    infobox.textprops["fontsize"] = fontsize
    infobox.titleprops["fontsize"] = fontsize
    infobox.draw(axes, loc)



def modparbox(self, axes=None, loc=3, title=None, fontsize=10, **kwargs):

    stringdict = OrderedDict()

    #formatfunc = str
    formatfunc = dashi.histfuncs.number_error_format

    for key in self.params:
        stringdict[key] = formatfunc(self.params[key], self.errors[key])

    kwargs['title'] = title
    infobox = InfoBox(stringdict, **kwargs) # need to pop kwargs
    infobox.textprops["fontsize"] = fontsize
    infobox.titleprops["fontsize"] = fontsize
    infobox.draw(axes, loc)


def p2dscatter(self, log=False, color=None, label=None, orientation='horizontal', **kwargs):
    """ use pylab.errorplot to visualize these scatter points
        
        Parameters:
          log        : if true create logartihmic plot

          (all other kwargs will be passed to pylab.errobar)
    """

    ax = p.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    
    kw = {"xerr" : self.xerr, "yerr" : self.yerr, "fmt" : "k", "capsize" : 0., "linestyle" : 'None', "color" : color}
    # for log-scaled axes, clip the lower extent of the error bar
    if log:
        kw["yerr"] = [n.where(self.y-self.yerr <= 0, abs(self.y)*(1-1e-12), self.yerr), self.yerr]
    kw.update(kwargs)

    if len(ax.lines) > 0:
        minvalue,maxvalue = p.ylim()
    else:
        minvalue,maxvalue = float('inf'),-float('inf')
    
    if orientation == 'vertical':
        set_xlim, set_ylim = ax.set_ylim, ax.set_xlim
        get_xlim, get_ylim = ax.get_ylim, ax.get_xlim
        set_yscale = ax.set_xscale
        x, y = self.y, self.x
        kw["xerr"], kw["yerr"] = kw["yerr"], kw["xerr"]
    else:
        set_xlim, set_ylim = ax.set_xlim, ax.set_ylim
        get_xlim, get_ylim = ax.get_xlim, ax.get_ylim
        set_yscale = ax.set_yscale
        x, y = self.x, self.y
    
    if len(self.x > 0):
        if not log:
            if n.abs(self.y).sum() > 0:
                minvalue = min( self.y.min(), minvalue )
                maxvalue = max( 1.3*self.y.max(), maxvalue)
                p.errorbar(x, y, **kw) 
            else:
                set_xlim(self.x[0], self.x[-1])
        else:
            set_yscale("log", nonposy='clip')
            ymi,yma = get_ylim()
            if n.abs(self.y).sum() > 0:
                minvalue = min(0.1 * self.y[self.y > 0].min(), ymi)
                maxvalue = max( n.power(10, n.ceil (n.log10(100. * self.y.max()))), maxvalue)
                p.errorbar(x, y, **kw) 
            else:
                set_xlim(self.x[0], self.x[-1])
        
        if not hasattr(ax, "_legend_proxy"):
            ax._legend_proxy = LegendProxy(ax)
        ax._legend_proxy.add_scatter(label=label, color=color)
        
        set_ylim(minvalue, maxvalue)
    _h2label(self, orientation)

def g2dimshow(self, **kwargs):
    """ plot the grid2d with pylab.imshow()
        Parameters:
          log    : if true create logartihmic plot. Empty
                   bins will be filled up the lowest populated decade.

          (all other kwargs will be passed to pylab.imshow)
    """
    z = self.z.T.copy()

    if kwargs.pop("log", False):
        if z.sum() > 0:
            #minval = 10**n.ceil(n.log10(bincontent[bincontent>0].min()))
            z[z==0] = n.nan # won't be drawn by imshow
            z = n.log10(z) # log of nans are nans
    else:
        if z.sum() > 0:
            z[z == 0] = n.nan
        
         

    kw = {"cmap": mpl.cm.jet, "aspect" : "auto", "interpolation" : "nearest" }
    kw.update(kwargs)
    p.imshow(z, origin="lower", 
             extent=(self.x[0], self.x[-1], self.y[0], self.y[-1]), **kw) 
    _h2label(self)

def g2dpcolor(self, **kwargs):
    """ plot the grid2d with pylab.pcolor()
        Parameters:
          log    : if true create logartihmic plot. Empty
                   bins will be filled up the lowest populated decade.

          (all other kwargs will be passed to pylab.pcolor)
    """
    bincontent = self.z.T.copy()
    x = n.concatenate((self.x-self.xerr, [self.x[-1]+self.xerr[-1]]))
    y = n.concatenate((self.y-self.yerr, [self.y[-1]+self.yerr[-1]]))
    
    _process_colorscale(bincontent, kwargs)
    return p.pcolor(x, y, bincontent, **kwargs)

def two_component_band(h1, h2, log=False, type="steps", cumulative=False, **kwargs):
    """ plot a filled band to indicate bincontents and binerrors 
    
        Parameters:
          log : if true create logartihmic plot
          cumulative : plot the cumulative histogram
          type: ['line', 'steps']

          (all other kwargs will be passed to pylab.fill_between)
    """
        
    both = h1 + h2
    both.band(log=log, type=type, cumulative=cumulative, facecolor=3*(0.95,), edgecolor=3*(0.15,) )
    h1.line(log=log, c="k", linestyle="dashed")
    both.line(log=log, c="k", linestyle="dashed")

def histcomp( histbdl, reference, cmphistolist, colorbdl, log=False, logratio=False, xlabel=None, ylabel=None):
    fig = p.figure()
    ax1 = p.axes([.15,.4,.8,.55])
    p.setp(ax1.get_xticklabels(), visible=False)
    ax2 = p.axes([.15,.1,.8,.3], sharex=ax1)

    p.gcf().sca(ax1)
    histbdl.line(c=colorbdl, log=log)
    p.gcf().sca(ax2)
    
    if ylabel is None:
        if logratio:
            ylabel = "log(ratio)"
        else:
            ylabel = "ratio"

    ratiomax = -n.inf
    ratiomin = +n.inf
    for key in cmphistolist:
        ratio = dashi.histfuncs.histratio(histbdl.get(key) , histbdl.get(reference), log=logratio, ylabel=ylabel)
        ratio.scatter(c=colorbdl.get(key))
        ratiomax = max( (ratio.y + ratio.yerr).max(), ratiomax)
        ratiomin = min( (ratio.y - ratio.yerr).min(), ratiomin)

    if xlabel is not None:
        ax2.set_xlabel(xlabel)

    if logratio:
        ratiomax = min(2, ratiomax)
        ratiomin = max(-2, ratiomin)
        ax2.set_ylim(ratiomin,ratiomax)
        p.axhline(0, c=colorbdl.get(reference), linestyle="dashed")
    else:
        ratiomax = min(5, ratiomax)
        ratiomin = 0
        ax2.set_ylim(ratiomin,ratiomax)
        p.axhline(1, c=colorbdl.get(reference), linestyle="dashed")
   
    p.gcf().canvas.draw()
    return ax1,ax2
