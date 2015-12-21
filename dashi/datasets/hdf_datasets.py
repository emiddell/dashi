"""
    wrappers around hdf files and hdf chains
    that provide the common dashi.datasets interface
"""

from dashi.datasets import Dataset
import os.path
import tables

PANDAS=False
if PANDAS:
    import pandas


def read_variable(h5file, path):
    """
        convenience function to read arrays from hdf files by specifying them
        with a path of the form:
         /my/path/to/table:column_name
        or 
         /my/path/to/an/array

        Parameters:
         h5file:    an open hdf file object
         path  :    a string containing the path
    """
            
    if ":" in path:
        path_, column = path.split(":")
        if PANDAS:
            return pandas.DataFrame(h5file.getNode(path_).col(column))
        return h5file.getNode(path_).col(column)
    else:
        if PANDAS:
            return pandas.DataFrame(h5file.getNode(path).read())
        return h5file.getNode(path).read()

################################################################################

def hdf_toc(h5file):
    """
        walk through the hdf file and create a table of content

        returns a list pathes: [ "/path/to/table:varname1", 
                                 "path/to/earray",
                                 ... ]
         
    """
    toc = []
    for node in h5file.walkNodes(classname="Table"):
        for varname in node.description._v_names:
            toc.append( "%s:%s" % (node._v_pathname, varname) )

    for arraytype in ["Array", "EArray"]:
        for node in h5file.walkNodes(classname=arraytype):
            toc.append( node._v_pathname ) 

    return sorted(toc)

################################################################################

 
def store_earray1d(h5file, path, array, complevel=6, **kwargs):
    """
        convenience function to create extendable arrays
    """

    parent = os.path.dirname(path)
    arrname = os.path.basename(path)

    filters = None
    if complevel > 0:
        filters = tables.Filters(complevel=complevel, complib="zlib")

    earr = h5file.createEArray(parent, arrname,  
                               tables.Atom.from_dtype(array.dtype), 
                               (0,),
                               filters=filters,
                               **kwargs)
    earr.append(array)
    earr.flush()

################################################################################

class HDFDataset(Dataset, tables.File):
    """
        wrapper around hdf files that can be attached to a hub
    """
    def __init__(self, filename, mode="r", expectedrows=1e6, filters=tables.Filters(complevel=6, complib="zlib"),
                 **kwargs):
        Dataset.__init__(self, os.path.basename(filename))
        tables.File.__init__(self, filename, mode, **kwargs)
        
        self._ds_expectedrows = expectedrows
        self._ds_filters = filters
        if mode == "r":
            self._ds_readonly = True
        else:
            self._ds_readonly = False

        self._ds_toc_cache = None

    @property
    def _ds_toc(self):
        if self._ds_toc_cache is None:
            toc = []
            if not self.isopen:
                return toc

            for node in self.walkNodes(classname="Table"):
                toc.append( node._v_pathname )
                for varname in node.description._v_names:
                    toc.append( "%s:%s" % (node._v_pathname, varname) )

            for node in self.walkNodes(classname="Array"):
                toc.append( node._v_pathname ) 

            self._ds_toc_cache = sorted(toc)

        return self._ds_toc_cache

    def _ds_read_variable(self, path):
        return read_variable(self, path)
    
    def _ds_write_variable(self, path, array):
        parent = os.path.dirname(path)
        arrname = os.path.basename(path)
        self._ds_toc_cache = None
        if array.dtype.names is None:
            earr = self.createEArray(parent, arrname,  
                                     tables.Atom.from_dtype(array.dtype), 
                                     (0,), filters=self.filters, createparents=True)
            earr.append(array)
            earr.flush()
        else:
            tab = self.createTable(parent, arrname, array, createparents=True, filters=self.filters)
            tab.flush()
    
    def _ds_remove_variable(self, path):
        try:
            self.removeNode(path)
            self.flush()
        except tables.NoSuchNodeError as exc:
            raise ValueError("removing path %s raised a NoSuchNodeError" % path)
        self._ds_toc_cache = None


################################################################################
try:
    from cubism.hdfchain import HDFChain
    class HDFChainDataset(Dataset, HDFChain):
        """
            wrapper around a chain of hdf files that can be attached to a hub
            /!\ read-only
        """
        def __init__(self, files, maxdepth=1, verbose=False, **kwargs):
            Dataset.__init__(self, "hdf chain <%s>" % str(files))
            HDFChain.__init__(self, files, maxdepth, verbose, **kwargs)
        
            self._ds_readonly = True
            self._ds_toc_cache = None

        @property
        def _ds_toc(self):
            if self._ds_toc_cache is None:
                toc = []

                for path, tableproxy in self.pathes.iteritems():
                    toc.append(path)
                    for varname in tableproxy._v_dtype.names:
                        toc.append("%s:%s" % (path, varname))

                self._ds_toc_cache = sorted(toc)

            return self._ds_toc_cache

        def _ds_read_variable(self, path):
            return read_variable(self, path)
except ImportError:
    pass
