
import dashi as d

def histsave(histo, file, where, name, overwrite=False):
    """
        store a histogram in a hdf file. It will create a group
        that contains the arrays of the histogram. Labels and
        necessary statistic information are stored as attributes
        of the group.

        file     : an open hdf file object
        where    : the parent group where the histogram should be saved
        name     : a name for the histogram
        overwrite: replace an existing stored histogram
    """
    import tables
    parentgroup = file.getNode(where)

    if name in parentgroup._v_children:
        if not overwrite:
            raise ValueError("there exists already a histogram with name %s" % name)
        else:
            file.removeNode(parentgroup, name, recursive=True)

    # create a new group and store all necessary arrays into it
    group = file.createGroup(where, name)
    attr = group._v_attrs

    attr["ndim"]  = histo.ndim
    attr["title"] = histo.title
    attr["nentries"] = histo.stats.values["nentries"]
    attr["nans"] = histo.stats.values["nans"]
    attr["nans_wgt"] = histo.stats.values["nans_wgt"]
    attr["nans_sqwgt"] = histo.stats.values["nans_sqwgt"]

    def save(arr, where):
        filters = tables.Filters(complib='blosc', complevel=9)
        ca = file.createCArray(group, where, tables.Atom.from_dtype(arr.dtype), arr.shape, filters=filters)
        ca[:] = arr

    save(histo._h_bincontent, "_h_bincontent")
    save(histo._h_squaredweights, "_h_squaredweights")
    
    # file.createArray(group, "_h_bincontent", histo._h_bincontent)
    # file.createArray(group, "_h_squaredweights", histo._h_squaredweights)
    for dim in range(histo.ndim):
        file.createArray(group, "_h_binedges_%d" % dim, histo._h_binedges[dim])
        attr["label_%d" % dim] = histo.labels[dim]


def histload(file, histgroup):
    """
        load from a hdf file a histogram that was stored with histsave

        file      : an open hdf file object
        histgroup : the group containing the histogram (the one created by histsave)
                    can be anything what file.getNode accepts, eg. a string or a Group 
                    object
    """
    import tables
    group = file.getNode(histgroup)
    attr = group._v_attrs
    histo = None
    
    ndim = attr["ndim"]

    binedges = [ group._v_children["_h_binedges_%d" % dim].read() for dim in range(ndim) ]
    labels   = [ attr["label_%d" % dim] for dim in range(ndim) ]

    kwargs = dict(bincontent=group._v_children["_h_bincontent"].read(),
        squaredweights=group._v_children["_h_squaredweights"].read(),
        title=attr["title"])

    if ndim == 1:
        histo = d.histogram.hist1d(binedges[0], label=labels[0], **kwargs)
    elif ndim == 2:
        histo = d.histogram.hist2d(binedges,labels=labels, **kwargs)
    else:
        histo = d.histogram.histogram(ndim, binedges,labels=labels, **kwargs)

    def load_attr(key, default):
        if key in attr:
            return attr[key]
        else:
            return default

    histo.stats.values["nentries"] = attr["nentries"] 
    histo.stats.values["nans"] = load_attr("nans", 0.) 
    histo.stats.values["nans_wgt"] = load_attr("nans_wgt", 0.)
    histo.stats.values["nans_sqwgt"] = load_attr("nans_sqwgt", 0.) 
    histo._h_newdataavailable = True

    return histo


