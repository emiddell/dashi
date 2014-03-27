
from dashi.odict import OrderedDict

class Dataset(object):
    """
        abstract base class to unify access to different input files, like hdf
        files, hdf chaines and maybe at some time also root files
    """

    def __init__(self, name):
        self._ds_name = name             # used for messages
        self._ds_readonly = False        # flag for readonly datasets

    @property
    def _ds_toc(self):
        raise NotImplementedError("must be overridden by subclass")

    def _ds_get(self, path):
        result = None
        if path in self._ds_toc:
            return  self._ds_read_variable(path) 
        else:
            raise ValueError("the variable with path %s is not contained in datafile %s" % (path, self._ds_name))

    def _ds_put(self, path, array):
        if self._ds_readonly:
            raise ValueError("this dataset is marked as readonly")
        else:
            self._ds_write_variable(path, array)

    def _ds_remove(self, path):
        if self._ds_readonly:
            raise ValueError("this dataset is marked as readonly")
        if not path in self._ds_toc:
            raise ValueError("the variable with path %s is not contained in datafile %s" % (path, self._ds_name))
        else:
            self._ds_remove_variable(path)

    def _ds_read_variable(self, path):
        raise NotImplementedError("must be overridden by subclass")
    def _ds_write_variable(self, path, array):
        raise NotImplementedError("must be overridden by subclass")
    def _ds_remove_variable(self, path):
        raise NotImplementedError("must be overridden by subclass")
