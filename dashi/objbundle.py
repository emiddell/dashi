
import operator
import copy as copymodule
import itertools
import threading
from Queue import Queue, Empty

bundle_classes = dict()

################################################################################

class object_bundle(object):
    """
        Base class for all object bundles.
    """
    def __init__(self, *args, **bdict):
        """
            construct a new object_bundle. Don't call this directly -
            use :meth:`dashi.objbundle.bundle`:: instead
        """
        object.__setattr__(self, "_b_objects", dict())
        self._b_type = None

        if len(args) != 0:
            if not isinstance(args[0], object_bundle):
                raise ValueError("only another bundle may be passed via a positional argument to __init__")
            if len(bdict) > 0 and any( [i in args[0].keys() for i in bdict.keys() ]):
                raise ValueError("overlap of keys in provided bundle and kwargs!")
            
            for name,obj in args[0]:
                self.add(name,obj)

        for name, obj in bdict.iteritems():
            self.add(name, obj)

    def __setattr__(self, name, value):
        if name in self._b_objects:
            self._b_objects[name] = value
            object.__setattr__(self, name, value)
        else:
            object.__setattr__(self, name, value)

    def add(self, name, obj):
        """
            Add an object to this bundle.

            **Parameters:**
              *name* : str 
                name of the object in the bundle
              *obj* : <the same type as all other objects in the bundle>
                the object to store in the bundle  
        """

        if self._b_type is None:
            self._b_type = obj.__class__
        else:
            if self._b_type != obj.__class__:
                raise ValueError("trying to add an object to a bundle with objects of different type!")

        if name in self._b_objects:
            raise ValueError("object with name %s is already contained in this bundle" % name)

        self._b_objects[name] = obj
        self.__dict__[name] = obj

    def __repr__(self):
        """
            pretty print this bundle
        """
        if len(self._b_objects) == 0:
            return "%s(empty)" % self.__class__.__name__
        else:
            maxlen = max( [len(i) for i in self._b_objects] )
            repr  = "%s(\n" % self.__class__.__name__
            fmt = " %" + str(maxlen+1) + "s=%s"
            repr += ",\n".join( [fmt  % (name, self._b_objects[name].__repr__()) for name in sorted(self._b_objects)]) 
            repr += "\n)"
            return repr

    # provide access to the _b_objects dictionary
    # __iter__ will behave like dict.iteritems
    def __iter__(self): return self._b_objects.iteritems()
    def keys(self): return sorted(self._b_objects.keys())
    def values(self): return [self._b_objects[i] for i in self.keys()]
    def get(self, name): return self._b_objects[name]
    def set(self, name,value): return self.__setattr__(name, value)

    def map(self, callable):
        """
            Map a callable to all values in this bundle.

            **Parameters:**
              *callable* : [function | object implementing __call__ ]
                The callable should expect one argument and will be executed
                for every object in the bundle.
        """
        return bundle(**dict([ (name, callable(obj)) for name,obj in self._b_objects.iteritems()]))
    
    def map_kv(self, callable):
        """
            Map a callable to all key-value pairs in this bundle.

            **Parameters:**
              *callable* : [function | object implementing __call__ ]
                The callable should expect two arguments and will be executed
                for every (key-value) pair in the bundle.
        """
        return bundle(**dict([ (name, callable(name, obj)) for name,obj in self._b_objects.iteritems()]))

    def __getattr__(self, name):
        """ 
            allow access to attributes of objects in the bundle. if "name" points to a
            non-callable, a bundle of the attribute values will be returned.
            If it is a callable, a 'bundleized' wrapper method is returned
        """
        attribute = None
        try:
            attribute = getattr(self._b_type, name)
        except AttributeError:
            raise ValueError("type %s has no attribute called %s" % (self._b_type.__name__, name))
    
        if callable(attribute):
            def wrapper(*args, **kwargs):
                bundles = []
                for arg in itertools.chain(args, kwargs.values()):
                    if isinstance(arg, object_bundle):
                        bundles.append(arg)

                # assert that bundles are compatible
                for bdl in bundles[1:]:
                    if bdl.keys() != bundles[0].keys():
                        raise ValueError("bundles have different keys")

                result = dict()
                for key in sorted(self._b_objects.keys()):
                    thisargs = []
                    thiskwargs = {}

                    for arg in args:
                        if isinstance(arg, object_bundle):
                            thisargs.append(arg.get(key))
                        else:
                            thisargs.append(arg)
                    
                    for kwarg_name,kwarg_value in kwargs.iteritems():
                        if isinstance(kwarg_value, object_bundle):
                            thiskwargs[kwarg_name] = kwarg_value.get(key)
                        else:
                            thiskwargs[kwarg_name] = kwarg_value

                    result[key] = attribute(self._b_objects[key],*thisargs, **thiskwargs)

                return bundle(**result)
    
            funcname = getattr(attribute, "func_name", attribute.__name__)

            wrapper.func_doc = "bundleized function '%s'" % funcname
            if hasattr(attribute, "func_doc") and (attribute.func_doc is not None):
                wrapper.func_doc += "\n" + attribute.func_doc
            return wrapper
        
        else:
            result = dict()
            for key in self.keys():
                result[key] = getattr(self._b_objects[key], name)

            return bundle(**result)
        


    def diversify(self, divdict, copy=False):
        """
            create a bundle with different keys from this bundle
            sometimes the keys of a bundle have to be modified to match
            the structure of a another bundle. the divdict defines
            how the elements of this bundle should be mapped to one ore
            more new keys. E.g.

            b1 = bundle(**{ "x" : 1, "y" : 2 })
            b2 = bundle(**{ "A" : 3, "B" : 4, "C" : 5})

            b3 = b1.diversify( { "x" : ["A"], "y" : ["B", "C"]} )
            creates the bundle
            b3 = bundle(**{ "A" : 1, "B" : 2, "C" : 2})

            if copy == False references are used to diversify the bundle
            if copy == True the elements of the bundle are copied to their new names

        """
        if sorted(divdict.keys()) != sorted(self.keys()):
            raise ValueError("the provided divdict an this bundle contain different keys")
        result = dict()
        for oldname, newnames in divdict.iteritems():
            for newname in newnames:
                if copy:
                    result[newname] = copymodule.copy( self.get(oldname) )
                else:
                    result[newname] = self.get(oldname)

        return bundle(**result)

    def select(self, keys):
        selection = dict( [ (key, self.get(key)) for key in keys ] )
        return bundle(**selection)

################################################################################

def _bundle_operator_generator(klass, methodname):
    def method(self, other):
        result = dict()
        if isinstance(other, object_bundle):
            if not sorted(self._b_objects.keys()) == sorted(other._b_objects.keys()):
                raise ValueError("object bundles are incompatible")
            for name, obj in self._b_objects.iteritems():
                #result[name] = getattr(obj,methodname)(other._b_objects[name]) 
                try:
                    result[name] = getattr(operator, methodname)(obj, other._b_objects[name])
                except TypeError:
                    result[name] = NotImplemented
        else:
            for name, obj in self._b_objects.iteritems():
                #result[name] = getattr(obj,methodname)(other)
                try:
                    result[name] = getattr(operator, methodname)(obj, other)
                except TypeError:
                    result[name] = NotImplemented

        # if any operation in a <op> b throws NotImplemented return a single NotImplemented
        # the interpreter may than try b <op> a
        if any([ i is NotImplemented for i in result.values()]):
            return NotImplemented
        else:
            return bundle_classes[klass](**result)

    return method

################################################################################

def _bundle_transpose(self):
    level1keys = self.keys()
    level1bundles = self.values()
    level2keys = []

    for key in level1bundles[0].keys():
        commonkey = True
        for bdl in level1bundles[1:]:
            if key not in bdl.keys():
                commonkey = False
        if commonkey:
            level2keys.append(key)

    result = dict()
    for l2key in level2keys:
        newbundle_dict = dict()
        for l1key in level1keys:
            newbundle_dict[l1key] = self.get(l1key).get(l2key)
        result[l2key] = bundle(**newbundle_dict)
    return bundle(**result)

################################################################################

def _bundle_generator(klass_instance):
    klass = klass_instance.__class__

    if klass in bundle_classes:
        return bundle_classes[klass]
    else:

        operator_names = ["__getitem__", "__lt__", "__le__", "__eq__", "__ne__",
                          "__gt__", "__ge__", "__add__", "__sub__", "__mul__",
                          "__floordiv__", "__mod__", "__divmod__", "__div__", "__truediv__", 
                          "__pow__", "__lshift__", "__rshift__", "__and__", "__xor__", 
                          "__or__", "__radd__", "__rsub__", "__rmul__", "__rdiv__", 
                          "__rtruediv__", "__rfloordiv__", "__rmod__", "__rdivmod__", 
                          "__rpow__", "__rlshift__", "__rrshift__",
                          "__rand__", "__rxor__", "__ror__"] 
        newclass_dict = dict()
        for opname in operator_names:
            #if hasattr(klass_instance, opname):
            #    newclass_dict[opname] = _bundle_operator_generator(klass, opname)
            newclass_dict[opname] = _bundle_operator_generator(klass, opname)

        if isinstance(klass_instance, object_bundle):
            newclass_dict["transpose"] = _bundle_transpose

        newclass_name = '%s_bundle' % klass.__name__
        newclass_type = type(newclass_name,(object_bundle,), newclass_dict)
        bundle_classes[klass] = newclass_type
        globals()[newclass_name] = newclass_type
        return newclass_type

################################################################################

def bundle(*args, **objects):
    """ 
        creates an object bundle
        pass objects by keywords parameters and/or provide another bundle
    """
    values = []
    if len(args) != 0:
        values.extend(args[0].values())

    values.extend( objects.values() )
    types = [i.__class__ for i in values]
    if not all([i == types[0] for i in types]):
        raise ValueError("all objects must be of the same type")
    return _bundle_generator(values[0])(*args,**objects)

################################################################################

def emptybundle(klass):
    return _bundle_generator(klass)()

################################################################################

def bundleize(method):
    """ 
        similar to numpy.vectorize this returns a wrapper around method
        which can operate on bundle arguments.
    """
    if not callable(method):
        raise ValueError("method must be callable")
    
    def wrapper(*args, **kwargs):
        bundles = []
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, object_bundle):
                bundles.append(arg)
        
        if len(bundles) == 0:
            return method(*args, **kwargs)

        for bdl in bundles[1:]:
            if bdl.keys() != bundles[0].keys():
                raise ValueError("bundles have different keys")

        result = dict()
        for key in bundles[0].keys():
            thisargs = []
            thiskwargs = {}

            for arg in args:
                if isinstance(arg, object_bundle):
                    thisargs.append(arg.get(key))
                else:
                    thisargs.append(arg)
            
            for kwarg_name,kwarg_value in kwargs.iteritems():
                if isinstance(kwarg_value, object_bundle):
                    thiskwargs[kwarg_name] = kwarg_value.get(key)
                else:
                    thiskwargs[kwarg_name] = kwarg_value

            result[key] = method(*thisargs, **thiskwargs)

        return bundle(**result)
    if hasattr(method, "func_name"):
        wrapper.func_doc = "bundleized function '%s'\n" % method.func_name
    else:
        wrapper.func_doc = "bundleized unnamed function\n"

    if hasattr(method, "func_doc") and (method.func_doc is not None):
        wrapper.func_doc += "\n" + method.func_doc

    return wrapper

################################################################################


def threaded_bundleize(method, nthreads):
    """
        threaded version of bundleize.
        FIXME: move this code into bundleize?
    """
    if not callable(method):
        raise ValueError("method must be callable")

    def wrapper(*args, **kwargs):
        bundles = []
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, object_bundle):
                bundles.append(arg)

        for bdl in bundles[1:]:
            if bdl.keys() != bundles[0].keys():
                raise ValueError("bundles have different keys")


        # setup queue, parse arguments and populate queue
        q = Queue()
        for key in bundles[0].keys():
            thisargs = []
            thiskwargs = {}

            for arg in args:
                if isinstance(arg, object_bundle):
                    thisargs.append(arg.get(key))
                else:
                    thisargs.append(arg)
            
            for kwarg_name,kwarg_value in kwargs.iteritems():
                if isinstance(kwarg_value, object_bundle):
                    thiskwargs[kwarg_name] = kwarg_value.get(key)
                else:
                    thiskwargs[kwarg_name] = kwarg_value

            q.put( (key, thisargs, thiskwargs) )
        
        # setup worker threads
        result = dict()
        def worker():
            keep_alive = True
            while keep_alive:
                try:
                    key, thisargs, thiskwargs = q.get(timeout=1.)
                    #print "thread %s working on key %s" % (threading.current_thread().name, key)

                    result[key] = method(*thisargs, **thiskwargs)
                    q.task_done()
                except Empty:
                    keep_alive = False

        for i in range(nthreads):
            t = threading.Thread(target=worker)
            t.daemon = True
            t.start()

        q.join() # wait until queue is empty                    

        return bundle(**result)

    wrapper.func_doc = "bundleized function '%s' for %d threads\n" % (method.func_name, nthreads)
    if hasattr(method, "func_doc") and (method.func_doc is not None):
        wrapper.func_doc += "\n" + method.func_doc

    return wrapper

################################################################################




