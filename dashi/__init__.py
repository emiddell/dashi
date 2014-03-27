"""
dashi documentations goes here
"""

import histfactory as factory
from fitting import model,gaussian,poly,leastsq #,poissonllh 
from visual import visual
from storage import histsave,histload
from objbundle import bundle, emptybundle, bundleize

import junkbox

import logging
logging.basicConfig(level=logging.INFO)

import tests
