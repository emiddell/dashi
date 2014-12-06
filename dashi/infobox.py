
import matplotlib as mpl
import pylab as p
import matplotlib.offsetbox

class InfoBox(object):
    def __init__(self, datadict, title = None, textprops = {"fontsize": 10, "family":"monospace"}, 
                 titleprops= {"fontsize":10, "weight":"bold", "family":"monospace"},
                 facecolor="w", edgecolor="k", alpha=1):
        self.title    = title
        self.datadict = datadict
        self.textprops = dict(textprops)
        self.titleprops = dict(titleprops)
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.alpha = alpha
        self.box = None


    def draw(self, axes=None, loc=2):
        if axes == None:
            axes = p.gca()

        if self.box != None and self.box in axes.artists:
            del axes.artists[ axes.artists.index(self.box) ]
        
        #key_areas = [ mpl.offsetbox.TextArea(k, textprops=self.textprops) for k in self.datadict.keys() ]
        #val_areas = [ mpl.offsetbox.TextArea(self.datadict[k], textprops=self.textprops) for k in self.datadict.keys() ]
        key_areas = [ mpl.offsetbox.TextArea(k, textprops=self.textprops) for k in list(self.datadict.keys()) ]
        val_areas = [ mpl.offsetbox.TextArea(self.datadict[k], textprops=self.textprops) for k in list(self.datadict.keys()) ]
        
        key_vpack = mpl.offsetbox.VPacker(children=key_areas, align="left", pad=0, sep=0)
        val_vpack = mpl.offsetbox.VPacker(children=val_areas, align="right", pad=0, sep=0)
        hpack = mpl.offsetbox.HPacker(children=[key_vpack, val_vpack], align="top", pad=0,sep=4)
        
        globchildren = []
        if self.title != None:
            #titlearea = mpl.offsetbox.TextArea(self.title, textprops=self.titleprops)
            titlearea = mpl.offsetbox.TextArea(self.title)
            globchildren.append(titlearea)
        
        globchildren.append(hpack)
        globvpack = mpl.offsetbox.VPacker(children=globchildren, align="center", pad=0, sep=1)
        
        self.box = mpl.offsetbox.AnchoredOffsetbox(loc=loc, child=globvpack)
        self.box.patch.set_facecolor(self.facecolor)
        self.box.patch.set_edgecolor(self.edgecolor)
        self.box.patch.set_alpha(self.alpha)
        axes.add_artist( self.box )

        if p.isinteractive():
            p.gcf().canvas.draw()
