.. dashi documentation master file, created by
   sphinx-quickstart on Thu Aug 26 09:09:41 2010.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===================
dashi documentation
===================

Elaborated data analyses are possible with the functionality offered by the great
`numpy <http://numpy.scipy.org/>`_, `matplotlib
<http://matplotlib.sourceforge.net/>`_, and `pytables
<http://www.pytables.org/moin>`_ libraries. However their support for
HEP-typical problems (histograms, fitting routines,visualization of those,..) is
limited. Dashi is intended to be a thin wrapper around these libraries to
provide some useful tools for these problems without obstructing the user the access
to the underlying libraries and without being a dependency sink.

Installation
============

The sources can be checked out from the IceCube SVN repository::
  
    http://code.icecube.wisc.edu/svn/sandbox/middell/dashi

After checkout to <topdir>/dashi make sure that <topdir> is in your PYTHONPATH. Much of
the functionality works if you have only numpy and matplotlib. For fitting
you will need to install `pyminuit2 <http://code.google.com/p/pyminuit2>`_. Some 
functionality requires `pytables <http://www.pytables.org/moin>`_. However, the library
will import these additional modules only when you try to use the functionality.

If you run into problems please let me know.

.. Tutorials:
    ==========

.. .. toctree::
   :maxdepth: 1

   histograms
   fitting
   data_handling

API Reference
=============

.. toctree::
   :maxdepth: 2

   api/index

Documentation
=============

.. toctree::
   :maxdepth: 1

   data_handling


Examples
========

.. toctree::
   :maxdepth: 1

   examples/hist1d
   examples/hist2d
   examples/fitting
   examples/fitting_hist

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

