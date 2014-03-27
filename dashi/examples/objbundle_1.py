
import numpy as n
import dashi as d

x = n.array( [1,2,3,4, 5] )
y = n.array( [6,7,8,9,10] )

bundle = d.bundle( x=x, y=y )

x.sum()
y.sum()

bundle.sum()
