################################################################################
# augment views on histogram classes
################################################################################

import dashi.histogram
import dashi.scatterpoints

def visual(enable=True):
    import dashi.histviews

    # hist1d augmentations 
    dashi.histogram.hist1d.line = dashi.histviews.h1line
    dashi.histogram.hist1d.scatter = dashi.histviews.h1scatter
    dashi.histogram.hist1d.band = dashi.histviews.h1band
    dashi.histogram.hist1d.statbox = dashi.histviews.h1statbox

    # hist2d augmentations
    dashi.histogram.hist2d.imshow  = dashi.histviews.h2imshow
    dashi.histogram.hist2d.pcolor  = dashi.histviews.h2pcolor
    dashi.histogram.hist2d.pcolormesh = dashi.histviews.h2pcolormesh
    dashi.histogram.hist2d.contour = dashi.histviews.h2contour
    dashi.histogram.hist2d.stack1d = dashi.histviews.h2stack1d

    # scatterpoints augmentations
    dashi.scatterpoints.points2d.scatter = dashi.histviews.p2dscatter 
    dashi.scatterpoints.grid2d.imshow = dashi.histviews.g2dimshow
    dashi.scatterpoints.grid2d.pcolor = dashi.histviews.g2dpcolor

    # other 
    dashi.fitting.model.parbox = dashi.histviews.modparbox

