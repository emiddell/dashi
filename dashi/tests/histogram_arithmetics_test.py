
import numpy as n
import dashi as d
from nose.tools import *

def test_add_histograms():
    bins = n.arange(-.5, 11.5)
    h1 = d.factory.hist1d( [1 , 1, 3, 10,10,10], bins)
    h2 = d.factory.hist1d( [1 , 4, 10,10], bins)

    assert (h1.bincontent == n.asarray([0,2,0,1,0,0,0,0,0,0,3])).all()
    assert (h2.bincontent == n.asarray([0,1,0,0,1,0,0,0,0,0,2])).all()

    assert ((h1+h2).bincontent == n.asarray([0,3,0,1,1,0,0,0,0,0,5])).all()
    assert ((h2+h1).bincontent == n.asarray([0,3,0,1,1,0,0,0,0,0,5])).all()

def test_multiply_histograms():
    bins = n.arange(-.5, 11.5)
    h1 = d.factory.hist1d( [1 , 1, 3, 10,10,10], bins)

    assert (h1.bincontent == n.asarray([0,2,0,1,0,0,0,0,0,0,3])).all()
    assert ((h1*2).bincontent == n.asarray([0,4,0,2,0,0,0,0,0,0,6])).all()

def test_divide_histograms():
    bins = n.arange(-.5, 11.5)
    h1 = d.factory.hist1d( [1 , 1, 3, 10,10,10], bins)

    assert (h1.bincontent == n.asarray([0,2,0,1,0,0,0,0,0,0,3])).all()
    assert ((h1/2.).bincontent == n.asarray([0,1,0,.5,0,0,0,0,0,0,1.5])).all()
