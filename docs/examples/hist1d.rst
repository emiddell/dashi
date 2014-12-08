.. _hist1dexample:

examples for 1d histograms
==========================

.. plot:: ../dashi/examples/hist1d.py
    :include-source:

.. plot::
    :include-source:

    import numpy as n
    import dashi as d
    d.visual()

    h = d.factory.hist1d( n.random.normal(2,2,1e5), n.linspace(-15,15,101), label="x")
    h.line()
    h.statbox()


.. plot::
    :include-source:

    import pylab as p
    import numpy as n
    import dashi as d
    d.visual()

    h1 = d.factory.hist1d( n.random.normal(2,2,1e5), n.linspace(-15,10,101), label=r"$\sqrt{x}$", title="Test")
    h1.line(filled=1, fc="r", alpha=.2)
    h2 = d.factory.hist1d( n.random.normal(-1,2,1e5), n.linspace(-15,10,101))
    h2.line(filled=1, fc="b", alpha=.2)
    h1.statbox(loc=1)
    h2.statbox(loc=2)
    p.ylim(0,10000)

