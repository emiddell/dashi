import os
from setuptools import setup

setup(
    name = "pydashi",
    version = "1.0.0",
    author = "The dashi developers",
    author_email = "eike@middell.net",
    description = ("TODO"),
    license = "LGPL",
    #keywords = "histograms fitting ",
    url = "https://github.com/emiddell/dashi",
    packages=['dashi', 'dashi.tests', 'dashi.datasets'],
    long_description='TODO',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python :: 2.7"
    ],
)
