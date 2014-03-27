
import numpy as n
import dashi as d
from nose.tools import *
import os

def test_bundle_creation():
    b = d.bundle(y=1, z=2, x=3)

    assert b.keys() == ["x", "y", "z"]
    assert b.x == 3
    assert b.y == 1
    assert b.z == 2

    assert b._b_type == int
    assert isinstance(b, d.objbundle.object_bundle)

def test_ndarray_bundle():
    x = n.random.normal(2,2,1e3)
    y = n.random.normal(-2,2,2e3)

    x_cut = x[x>0]
    y_cut = y[y>0]

    b = d.bundle(x=x,y=y)
    b_cut = b[b>0]
    assert (b_cut.x == x_cut).all()
    assert (b_cut.y == y_cut).all()

    b_add = b + 2
    assert (b_add.x == (x+2)).all()
    assert (b_add.y == (y+2)).all()

    shapes = b_cut.shape
    assert shapes.x == x_cut.shape
    assert shapes.y == y_cut.shape

    sums = b_cut.sum()
    assert sums.x == x_cut.sum()
    assert sums.y == y_cut.sum()


def test_int_bundle():
    x = 5 
    y = -2

    x_mask = x>0
    y_mask = y>0

    b = d.bundle(x=x,y=y)
    b_mask = b>0
    assert b_mask.x == x_mask
    assert b_mask.y == y_mask

    b_add = b + 2
    assert b_add.x == (x+2)
    assert b_add.y == (y+2)

def test_diversify():
    x = n.random.normal(2,2,1e3)
    y = n.random.normal(-2,2,2e3)

    b = d.bundle(x=x,y=y)

    b2 = b.diversify({ "x" : ["x1", "x2"], "y" : ["y1", "y2"]})

    assert b2.keys() == ["x1", "x2", "y1", "y2"]
    assert id(b2.x1) == id(x)
    assert id(b2.x2) == id(x)
    assert id(b2.y1) == id(y)
    assert id(b2.y2) == id(y)

    b3 = b.diversify({ "x" : ["x1", "x2"], "y" : ["y1", "y2"]}, copy=True)
    assert id(b3.x1) != id(x)
    assert id(b3.x2) != id(x)
    assert id(b3.y1) != id(y)
    assert id(b3.y2) != id(y)
    assert (b3.x1 == x).all()
    assert (b3.x2 == x).all()
    assert (b3.y1 == y).all()
    assert (b3.y2 == y).all()


def test_emptybundle():
    b = d.emptybundle(n.ndarray)
    b.add("x", n.random.normal(2,2,1e3))
    b.add("y", n.random.normal(-2,2,1e3))

    mean = b.mean()

def test_transpose():
    nelements = 1e3
    cat_a = d.bundle(x=n.array([1,2,3]), y=n.array([4,5,6]), z=n.array([7,8,9]))
    cat_b = d.bundle(x=n.array([1,2,3]), y=n.array([4,5,6]), z=n.array([7,8,9]))
    cat_c = d.bundle(x=n.array([1,2,3]), y=n.array([4,5,6]), z=n.array([7,8,9]), other=n.array([10,11,12]))

    b = d.bundle(cat_a=cat_a, cat_b=cat_b, cat_c=cat_c)

    assert all(b.cat_a.x == [1,2,3])
    assert all(b.cat_b.y == [4,5,6])
    assert all(b.cat_c.y == [4,5,6])

    assert b.cat_a.keys() == ["x","y","z"]
    assert b.cat_b.keys() == ["x","y","z"]
    assert b.cat_c.keys() == ["other", "x","y","z"]
    
    bt = b.transpose()
    
    assert bt.x.keys() == ["cat_a", "cat_b", "cat_c"]
    assert bt.y.keys() == ["cat_a", "cat_b", "cat_c"]
    assert bt.z.keys() == ["cat_a", "cat_b", "cat_c"]
    assert "other" not in bt.keys()

    assert id(bt.x.cat_a) == id(b.cat_a.x)
    assert id(bt.y.cat_a) == id(b.cat_a.y)
    assert id(bt.z.cat_a) == id(b.cat_a.z)


