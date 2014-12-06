import numpy as n
import scipy.stats
import inspect
import copy
import sys
from dashi.odict import OrderedDict
from logging import getLogger 


class model(object):
    """
        This class represents a model that can be fitted to data. It is 
        a wrapper around a callable object that gives names to the parameters 
        and can hold the fit results.

        the callable must be a function with signature f(x, p1, p2,...)

        :param params: bla
        :type params: dict

    """

    def __init__(self, callable, integral_callable=None):
        """
        :params callable: a function dP/dx
        :params integral_callable: a function P(X < x)
        """
        self.dfunc = callable
        argspec = inspect.getargspec(callable)
        self.func = integral_callable

        # initialize params with 1 which is a better starting value than zero for factors
        self.params = OrderedDict( [(i, 1.) for i in argspec.args[1:]] )
        self.errors = OrderedDict( [(i, 0.) for i in argspec.args[1:]] )
        self.limits = OrderedDict( [(i, None) for i in argspec.args[1:]] )
        self.cov    = n.ones( (len(self.params), len(self.params) ), dtype=float) * n.nan
        self.fixed  = set()
        self._integral = False

    @property
    def differential(self):
        if self._integral:
            inst = copy.copy(self)
            inst._integral = False
            return inst
        else:
            return self
    
    @property
    def integral(self):
        if not self._integral:
            inst = copy.copy(self)
            inst._integral = True
            return inst
        else:
            return self

    def __call__(self, *args, **kwargs):
        if len(args) == 0:
            raise ValueError("provide at least x")
        if len(args) == 1 and len(kwargs) == 0:
            getLogger("dashi.fitting").info("use current parameter values") 
            kwargs.update(self.params)
        # add fixed parameters
        for k in self.fixed:
            if not k in kwargs:
                kwargs[k] = self.params[k]
        if self._integral:
            if self.func is None:
                raise NotImplementedError("%s has no integral form" % type(self))
            return self.func(*args, **kwargs)
        else:
            return self.dfunc(*args, **kwargs)

    def first_guess(self, x, data):
        """
            called by the fitting routine. allows the model
            to roughly estimate the parameters from the data
            to provide a seed for the minimizer
        """
        pass

class gaussian(model):
    """ normal distribution """
    def __init__(self):
        f = lambda x, amp, mean,sigma :  amp * scipy.stats.norm.cdf(x, loc=mean, scale=sigma)
        df = lambda x, amp, mean,sigma :  amp * scipy.stats.norm.pdf(x, loc=mean, scale=sigma)
        model.__init__(self, df, f)

    def first_guess(self, x, data, mask=None):

        self.params["mean"] = (x*data).sum() / data.sum()

        meansquared = (x**2 * data).sum() / data.sum()
        self.params["sigma"] = n.sqrt( meansquared - self.params["mean"]**2 ) 
        # Minuit doesn't like 1-sided limits
        self.limits["sigma"] = (0, self.params["sigma"]*1e6)

        if mask is None:
            mask = (x > (self.params['mean'] - 3*self.params['sigma']))&(x < (self.params['mean'] + 3*self.params['sigma']))
            self.first_guess(x[mask], data[mask], mask)
        
        self.params["amp"] = data.max()*self.params["sigma"]*n.sqrt(n.pi*2.)

        self.errors["amp"]   =  0.5 * self.params["amp"]
        self.errors["mean"]  =  0.5 * self.params["sigma"]
        self.errors["sigma"] =  0.5 * self.params["sigma"]

class poly(model):
    def __init__(self, degree):
        assert degree >= 0

        if degree == 0:
            df = lambda x, p0 : p0
        elif degree == 1:
            df = lambda x, p0,p1 : p0 + p1*x
        else:
            dfuncstr = "df = lambda x," + ",".join(["p%d" % i for i in range(degree+1)]) + " : "
            funcstr = dfuncstr[1:]
            funcstr += "p1 + " + " + ".join(["(p%d/%.1f)*x**%d" % (i, i+1,i+1) for i in range(2,degree+1)])
            dfuncstr += "p0 + p1*x + " + " + ".join(["p%d*x**%d" % (i,i) for i in range(2,degree+1)])
            exec(dfuncstr)
            exec(funcstr)

        model.__init__(self, df, f)

    def first_guess(self, x, data):
        pass

class powerlaw(model):
    r"""
    A power-law of the form:
    
    .. math::
        
        dN/dx = N \frac{-\gamma - 1}{x_{min}} \frac{x}{x_{min}}**\gamma
        
    where :math:`x_{min} > 0` and :math:`\gamma \lt -1`
    """
    def __init__(self):
        df = lambda x, norm, index, pivot : norm * (-(1+index)/pivot)*(x/pivot)**index
        f = lambda x, norm, index, pivot: norm * (1 - (x/pivot)**(index+1))
        model.__init__(self, df, f)
        self.params["pivot"] = 1
        self.fixed.add("pivot")
        self.limits["index"] = (-1e6, -1)
    
    def first_guess(self, x, data):
        self.params["pivot"] = x.min()
        self.params["index"] = -(1+ data.sum()/sum(data*n.log(x/self.params["pivot"])))
        self.params["norm"] = data.sum()        

def _bfgs(model, chi2, verbose=True):
    """
    Adapter for low-memory BFGS with bounds
    """
    
    import inspect
    free = [k for k in list(model.params.keys()) if not k in model.fixed]
    def chi2f(args):
        kwargs = dict(list(zip(free, args)))
        return chi2(**kwargs)
    
    x0 = []
    bounds = []
    for k in model.params:
        if k in model.fixed:
            continue
        x0.append(model.params[k])
        v = model.limits[k]
        if v is None:
            bounds.append((None, None))
        elif isinstance(v, tuple):
            bounds.append(v)
        else:
            raise TypeError("Bounds must be specified as a tuple of floats (got %s)" % v)
    
    x, fval, info = optimize.fmin_l_bfgs_b(chi2f, x0, bounds=bounds, approx_grad=True)
    
    if info['warnflag'] == 1:
         getLogger("dashi.fitting").error("Failed to converge: too many function evaluations (%d)" % info['funcalls'])
    elif info['warnflag'] == 2:
         getLogger("dashi.fitting").error("Failed to converge: %s" % info['task'])
         raise RuntimeError("Failed to converge: %s" % info['task'])
    
    # store results and return model
    for i, k in enumerate(free):
        model.params[k] = x[i]
    # NB: BFGS doesn't calculate errors
    model.chi2 = fval
    
    return model

def _minuit(model, chi2, verbose=True):
    """
    Adapter for Minuit
    """
    # setup minuit
    setupdict = dict(model.params)
    for key in model.params:
        if key in model.fixed:
            del setupdict[key]
            continue
        if model.errors[key] != 0:
            setupdict["err_%s" % key] = model.errors[key]
        lim = model.limits[key]
        if lim != None:
            if type(lim) != tuple:
                raise ValueError("limits must be specified as a tuple of floats")
            if lim[0] is None and lim[1] is None:
                continue
            setupdict["limit_%s" % key] = lim
    
    minuit = minuit2.Minuit2(chi2, **setupdict)
    if verbose:
        minuit.printMode=1

    # TODO set minuit.up to 1 or .5 

    # perform minimization and error estimation
    try:
        minuit.migrad()
        minuit.hesse()
    except minuit2.MinuitError as exc:
        getLogger("dashi.fitting").error("minuit error catched: %s" % str(exc))
        
    # store results and return model
    model.params.update(minuit.values)
    model.errors.update(minuit.errors)
    model.cov  = n.asarray( minuit.matrix() )
    model.chi2 = minuit.fval

_minimize = None
_minimizers = list()
try:
    from scipy import optimize
    _minimize = _bfgs
    _minimizers.append(_bfgs)
except ImportError:
    pass
try:
    import minuit2
    _minimize = _minuit
    _minimizers.append(_minuit)
except ImportError:
    pass

def _prepare_data(x, data, error, integral):
    x = n.asarray(x)
    data = n.asarray(data)
    if error!=None:
        error = n.asarray(error)
    if integral:
        centers = 0.5*(x[1:]+x[:-1])
    else:
        centers = x
    
    return x, centers, data, error

def exec_(stmt, locals=None):
    if sys.version_info.major > 2:
        exec(stmt, locals)
    else:
        exec("""exec stmt in locals""")
    return locals

def _define_objfun(expr, x, data, weight, model, integral):
    """
    :param expr: expression for the element-wise contribution to the objective function
    """
    from numpy import diff, log
    free = [k for k in list(model.params.keys()) if k not in model.fixed]
    varstring = ",".join(free)
    mod = "model(x, %s)" % (",".join(["%s=%s" % (k,k) for k in free]))
    
    if integral:
        model = model.integral
        mod = "diff(%s)" % mod
    expr = expr.replace('model', mod)
    funcdef = 'chi2 = lambda %(varstring)s: (%(expr)s).sum()' % locals()
    funcdef_values = 'chi2valfunc = lambda %(varstring)s: %(expr)s' % locals()
    ldict = exec_(funcdef, locals())
    chi2 = ldict['chi2']
    ldict = exec_(funcdef_values, locals())
    chi2valfunc = ldict['chi2valfunc']
    return chi2, chi2valfunc

def leastsq(x, data, model, error=None, integral=False, verbose=True, chi2values=False, first_guess=True):
    """
    Perform a least squares fit of a model to data points given by data and x.
        
    **Parameters**:
        *x* : numpy.ndarray
            list of data points (x-values)
        *data* : numpy.ndarray
            list of data points (y-values)
        *model* : :class:`~dashi.fitting.model` object
            the model that should be fitted to the data
        *error* : [ None | numpy.ndarray ]
            error values for the points given by *x* and *data*
        *verbose* : [ True | False ]
            generate more output
        *chi2values* : [ True | False ]
            store for each point in *x* and *data* the individual contribution to the chi2
        *first_guess* : [True | False ]
            if true the models :meth:`~dashi.fitting.model.first_guess` method is called. 
            Oterwise the minimizer will start at the values already stored in
            :attr:`~dashi.fitting.model.params`
        
    **Return Value**:         
        the passed in :class:`~dashi.fitting.model` object with the fitted parameters 
    """

    if _minimize is None:
        getLogger("dashi.fitting").error("You need to install pyminuit2 or scipy")
        raise RuntimeError('No minimizers found!')
    
    x, centers, data, error = _prepare_data(x, data, error, integral)
    
    # define residual between data and model
    # use exec to create a proper argument list that can be parsed by inspect (needed by minuit)

    if error is None:
        weight = 1.
    else:
        weight = 1./error**2
        weight[~n.isfinite(weight)] = 0
    chi2, chi2valfunc = _define_objfun("weight*(data - model)**2", x, data, weight, model, integral)
    
    # give the model a chance to have a first look on the data to set the starting values
    if first_guess:
        model.first_guess(centers, data)
        
    # return model
    _minimize(model, chi2, verbose)

    model.ndof = len(data)-len(model.params)
    model.chi2prob = scipy.stats.chisqprob(model.chi2, model.ndof)

    if chi2values:
        model.chi2values = centers, chi2valfunc(*(v for k, v in list(model.params.items()) if not k in model.fixed))

    return model


def poissonllh(x, data, model, error=None, integral=True, first_guess=True, verbose=True):
    """
    Poisson log liklihood fit
    
    Model must be of type "model"
    
    Params:
        x - list of data points (x-values)
        data - list of data points (y-values)
        model - model(x, p1, p2, ...)
        error - a list of error values of points given in data
        verbose - more output
        
    Return:         
        the passed in model object with the fitted parameters 

    """
    
    if _minimize is None:
        getLogger("dashi.fitting").error("You need to install pyminuit2 or scipy")
        raise RuntimeError('No minimizers found!')
    
    x, centers, data, error = _prepare_data(x, data, None, integral)
    
    # define residual between data and model
    # use exec to create a proper argument list that can be parsed by inspect (needed by minuit)
    weight = None
    llh, valfunc = _define_objfun("model - data*log(model)", x, data, weight, model, integral)
    
    # give the model a chance to have a first look on the data to set the starting values
    if first_guess:
        model.first_guess(centers, data)
        
    # return model
    _minimize(model, llh, verbose)

    return model
    
    
    try:
        import minuit2
    except ImportError:
        getLogger("dashi.fitting").error("You need to install pyminuit2")
        raise

    print()
    print("FIXME There several changes that were only made to leastsq -> check!")
    print()
    1/0 # bail out

    # define residual between data and model
    # use exec to create a proper argument list that can be parsed by inspect (needed by minuit)

    funcdef = None
    #varstring = ",".join(model.parnames)
    varstring = ",".join(list(model.params.keys()))
    log = n.log
    # -log L = mu - n * log mu + ( log (n!) )   | combinatorial term can be omitted
    if error is None:
        chi_funcdef = "chi2 = lambda %s: ((data - model(x, %s))**2).sum()" % (varstring, varstring)
    else:
        chi_funcdef = "chi2 = lambda %s: (((data - model(x, %s))/ error)**2).sum()" % (varstring, varstring) 
    exec(chi_funcdef, locals())
    
    llh_funcdef = "llh = lambda %s: (model(x, %s) - data*log(model(x,%s))).sum()" % (varstring, varstring, varstring)
    exec(llh_funcdef, locals())

    # give the model a chance to have a first look on the data to set the starting values
    model.first_guess(x, data)
    
    # setup minuit
    minuit = minuit2.Minuit2(llh, **model.params)
    if verbose:
        minuit.printMode=1
    
    # TODO set minuit.up to 1 or .5 

    # perform minimization and error estimation
    minuit.migrad()
    minuit.hesse()

    # store results and return model
    model.params.update(minuit.values)
    model.errors.update(minuit.errors)
    model.llh  = minuit.fval

    # calculate also chi2 
    model.chi2 = chi2(**model.params)
    model.ndof = len(data)-len(model.params)
    model.chi2prob = scipy.stats.chisqprob(model.chi2, model.ndof)

    return model
