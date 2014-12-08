dashi
=====

Elaborate data analyses are possible with the functionality offered by the
great [numpy](http://numpy.scipy.org/), [matplotlib](http://matplotlib.org),
and [pytables](http://www.pytables.org/moin) libraries. However, their support
for HEP-typical problems (histograms, fitting routines, visualization of those,
etc) is limited. Dashi is intended to be a thin wrapper around these libraries
to provide some useful tools for these problems without obstructing the user
the access to the underlying libraries and without being a dependency sink.

# Installation

The easiest way to install dashi is with `pip`:

	pip install https://github.com/emiddell/dashi/zipball/master
