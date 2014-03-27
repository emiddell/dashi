
import dashi as d
#from dashi.datasets.dataset import HDFDataset, HDFChainDataset
import numpy as n
import new
from collections import defaultdict
import time
import sys

class DatasetHub(object):
    """
        several datasets can be connected to a hub which then 
        allows a retrieve objbundles of variables
    """
    def __init__(self):
        self.datasets = dict()    
        self.vars = dict()

    def __del__(self):
        for k in self.datasets:
            self.datasets[k] = None

    def connect_dataset(self, name, dataset):
        dataset._ds_hub = self
        self.datasets[name] = dataset
        self.__dict__[name] = dataset

    def disconnect_dataset(self, name):
        if name in self.datasets:
            del self.__dict__[name]
            del self.datasets[name]

    def get(self, vars, unpack_recarrays=False):
        """
            varname is either a string or a list of strings 
            with variable names


            returns either a ndarray_bundle or a ndarray_bundle_bundle
        """
        start = time.time()

        def get_one_variable(self,varname,current,total, unpack_recarrays=False):
            " helper function that retrieves a single variable"
            print "  %3d/%d reading variable %s" % (current,total,varname),
            start2 = time.time()
            arrays = {}
            missing_datasets = []
            for name,dataset in self.datasets.iteritems():
                tmp = None
                try:
                    if varname in self.vars and (self.vars[varname].vardef is not None):
                        v = self.vars[varname]
                        tmp = dataset._ds_get(v.vardef)
                        if v.transform is not None:
                            tmp = v.transform(tmp)
                    else:
                        tmp = dataset._ds_get(varname)
                except ValueError:
                    missing_datasets.append(name)

                # tmp is now pointing either to None, a 1d array or a recarray with named columns
                if tmp is not None:
                    # unpack the different columns of the recarray into 1d arrays in differnt
                    # slots of the resulting bundle
                    if unpack_recarrays: 
                        if tmp.dtype.names is None:
                            arrays[name] = tmp
                        else:
                            for column in tmp.dtype.names:
                                arrays[name+"_"+column] = tmp[column]

                    # just store the array
                    else:
                        arrays[name] = tmp

            if len(arrays) == 0:
                print "| done after %d seconds" % (time.time() - start2)
                return None
            
            # add empty arrays where necessary
            # rationale: empty arrays are easier to handle than bundles with missing keys
            # TODO: maybe make this configureable
            if len(missing_datasets) > 0:  
                dtype = arrays.values()[0].dtype
                for name in missing_datasets:
                    arrays[name] = n.zeros(0, dtype=dtype)
                print "| filling empty keys",
            print "| done after %d seconds" % (time.time() - start2)
            sys.stdout.flush()

            return d.bundle(**arrays)


        if isinstance(vars, str):
            tmp = get_one_variable(self, vars, 1,1, unpack_recarrays)
            print "total time:", time.time()-start
            return tmp
        elif isinstance(vars, list) and all([isinstance(i, str) for i in vars]):
            bundles = dict( [ (varname, get_one_variable(self, varname,i+1,len(vars),unpack_recarrays)) 
                              for i,varname in enumerate(vars)] )
            bundles = dict( [ (i,j) for i,j in bundles.iteritems() if j is not None ] )
            if len(bundles) == 0:
                print "total time:", time.time()-start
                return None
            else:
                tmp =  d.bundle(**bundles)
                print "total time:", time.time()-start
                return tmp
        else:
            raise ValueError("vars must be either a string or a list of strings")

    def put(self, path, bdl):
        for key in bdl.keys():
            if not key in self.datasets:
                raise ValueError("this bundle contains key %s which corresponds to no connected dataset")

        for key in bdl.keys():
            self.datasets[key]._ds_put(path, bdl.get(key))

    def remove(self, path):
        for key, ds in self.datasets.iteritems():
            errors = []
            try:
                ds._ds_remove_variable(path)
            except ValueError as exc:
                errors += [key]

        if len(errors) == len(self.datasets):
            raise ValueError("while removing '%s' got errors from _all_ datasets!" % path)
        elif (0 < len(errors)) and ( len(errors) < len(self.datasets)):
            print "caught errors while removing '%s' for datasets %s" % (path, " ".join(errors))

            

    def keys(self):
        return self.datasets.keys()


    def print_toc(self, varsonly=False):
        """
            print a list of all available variables in this hub together
            with a flag in which of the connected datasets the are
            available
        """
        global_toc = defaultdict(set)
        maxpathlen = 0
        maxdsnamelen = 0
        for dsname in self.datasets:
            maxdsnamelen = max(len(dsname), maxdsnamelen)
            thistoc = self.datasets[dsname]._ds_toc

            if varsonly:
                for vname,v in self.vars.iteritems():
                    print vname,v
                    if v.vardef in thistoc:
                        global_toc[vname].add(dsname)
                        maxpathlen = max(len(vname), maxpathlen)
            else:
                for path in thistoc:
                    global_toc[path].add(dsname)
                    maxpathlen = max(len(path), maxpathlen)
        fmt_substring = lambda size : "%"+str(size)+"s"

        fmt = fmt_substring(maxpathlen)
        totsize = maxpathlen
        keys = sorted(self.datasets.keys())
        for dsname in keys:
            fmt += " " + fmt_substring(len(dsname))
            totsize += len(dsname) + 1
    
        print fmt % tuple( [""] + keys)
        print totsize * "-"
        for path in sorted(global_toc.keys()):
            def marker(k):
                if k in global_toc[path]:
                    return "x"
                else:
                    return "o"
            print fmt % tuple( [path] + [marker(key) for key in keys])

    def get_vars(self, vardef_only=True):
        """
            return the keys of the registered variables

            If vardef_only is True only those variable keys are returned
            for which a vardef is defined (e.g. a path in a hdf file is specified).
            In this case the keys are returned sort by their vardefs.
        """
        if not vardef_only:
            return self.vars.keys()
        else:
            varcmp = lambda v1,v2 : cmp(self.vars[v1].vardef, self.vars[v2].vardef) # group by hdf tables, may improve speed
            readable =  [ i for i in self.vars.keys() if self.vars[i].vardef is not None ]
            return sorted(readable, cmp=varcmp) 

def usermethod(obj):
    """
        wrapper that allows to attach methods to a hub
    """
    def wrapper(method):
        setattr(obj, method.func_name, new.instancemethod(method, obj, obj.__class__))

    return wrapper
