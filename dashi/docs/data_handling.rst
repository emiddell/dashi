==============
Object Bundles
==============

Object bundles are intended to simplify the work with sets of similar objects.
They are basically dictionaries with the additional feature, that method calls 
on the bundle are executed on each object in the bundle. 

.. literalinclude:: ../examples/objbundle_1.py
    :language: python
    :linenos:



Numpy arrays are a very useful container for the data that one wants to analyse.
Because of the element-wise operation of operators and functions calculations
are easily formulated and fast. Hence, most of my analysis scripts hold the data
in memory and stored in numpy arrays. Multiple variables or classes of measurements 
can be distinguished, by keeping the arrays in dictionaries

Object bundles are intended to solve two problems:
 * 


The element-wise operation of operators and functions on 
numpy arrays make them a very handy tool for calculations. For example
and can add two arrays and the operation will add the respective
items from each array.

One can reapply this concept one level higher and apply 
can add two 
one can e.g add two numbers 


