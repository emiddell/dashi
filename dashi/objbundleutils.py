
import numpy as n
from objbundle import bundle
import inspect

def ndarray2bundle(array, varselection=None):
    if not isinstance(array, n.ndarray):
        raise ValueError("array must be of type numpy.ndarray")

    names = array.dtype.names
    if varselection is not None:
        missing = [i for i in varselection if i not in names]
        if len(missing) > 0:
            raise ValueError("this array has no variable(s) called %s" % ( " ".join(missing) ))
        else:
            return bundle( **dict([ (i, array[i]) for i in varselection]))
    else:
        return bundle( **dict([ (i, array[i]) for i in names ]))

def bundle2ndarray(bdl, varselection=None):
    if bdl._b_type != n.ndarray:
        raise ValueError("bdl must of type ndarray_bundle")
    
    dtype = None
    if varselection is not None:
        missing = [i for i in varselection if i not in bdl.keys()]
        if len(missing) > 0:
            raise ValueError("this bundle has no variable(s) called %s" % ( " ".join(missing) ))
        else:
            dtype = [ (i,bdl._b_objects[i].dtype) for i in varselection ]
    else:
        dtype = [ (i,bdl._b_objects[i].dtype) for i in bdl.keys() ]

    vars = [ i[0] for i in dtype]
    lengths = n.unique( [ len(bdl.get(i)) for i in vars ])
    if len(lengths) > 1:
        raise ValueError("(selected) arrays in bdl must have unique lengths")

    result = n.zeros(lengths[0], dtype=dtype)
    for i in vars:
        result[i] = bdl.get(i)

    return result

def read_hdf(h5file, selection):
    """
        read a selection of columns in this hdf file
        h5file is an open pytables file object
        selection is a dict of a form like e.g.
          { "varname"  : "/path/to/table:columname",
            "varname2" : lambda file: n.log10(file.root.mytable.col("var")),
            "var3"     : lambda varname, varname2 : 2*varname + varname2}

        returns a ndarray_bundle

    """
    # cfg checks. determine read order, move variable without dependencies
    # to the beginning, variables with dependencies either to the end
    # or before the last variable that depends on them
    readorder = [] # list of tuples [(varname, [depvar1, depvar2]) ]
    while len(readorder) != len(selection):
        vars_in_readorder = [i[0] for i in readorder]
        for varname, cfg in selection.iteritems():
            if varname in vars_in_readorder:
                continue
            if isinstance(cfg, str):
                # path, doesn't depend on anything. put to the beginning of the list
                readorder.insert(0, (varname, [] ))
                #print "insert", varname
            elif callable(cfg):
                args = inspect.getargspec(cfg).args
                if args == ["file"]:
                    # pass h5file, calculation doesn't depend on anything
                    readorder.insert(0, (varname, []) )
                    #print "insert", varname
                elif all(map(lambda i : i in selection, args)):
                    #print "considering", varname
                    if all( [i in vars_in_readorder for i in args]):
                        # all deps are there. look for insert position
                        for idx in range(len(readorder)+1):
                            varnames_before = [i[0] for i in readorder[:idx]]
                            if all([i in varnames_before for i in args]):
                                #print "insert", varname
                                readorder.insert(idx, (varname, args))
                                break
                    else:
                        continue
                else:
                    raise ValueError("arguments of callable for varname %s are unknown" % varname)
    #print ("reading %d variables in the following order: " % len(readorder))+ " ".join([i[0] for i in readorder])

    arrays = dict()
    for varname,deps  in readorder:
        cfg = selection[varname]
        if isinstance(cfg, str):
            if ":" in cfg:
                path, column = cfg.split(":")
                arrays[varname] = h5file.getNode(path).col(column)
            else:
                arrays[varname] = h5file.getNode(cfg).read()
        elif callable(cfg):
            args = inspect.getargspec(cfg).args
            if args == ["file"]:
                arrays[varname] = cfg(h5file)
            else:
                arrays[varname] = cfg(*[arrays[i] for i in args])

    return bundle(**arrays)


def bundle2h5group(bundle, file, h5group):
    import tables
    assert isinstance(h5group, str)
    for key,array in bundle:
        earr = file.createEArray(h5group, key,  
                                 tables.Atom.from_dtype(array.dtype), 
                                 (0,), filters=tables.Filters(complevel=6, complib="zlib"), createparents=True)
        earr.append(array)
        earr.flush()

def h5group2bundle(file, h5group):
    import tables
    assert isinstance(h5group, str)
    arrays = dict()

    for key in file.getNode(h5group)._v_children.keys():
        arrays[key] = file.getNode(h5group+"/"+key).read()

    return bundle(**arrays)
