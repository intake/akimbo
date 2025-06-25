# akimbo

**For when your data won't fit in your dataframe**

[![Tests](https://github.com/intake/akimbo/actions/workflows/pypi.yml/badge.svg)](https://github.com/intake/akimbo/actions/workflows/pypi.yml)
[![Documentation Status](https://readthedocs.org/projects/akimbo/badge/?version=latest)](https://akimbo.readthedocs.io/en/latest/?badge=latest)

Akimbo provides fast, vectorized processing of nested, ragged data
in dataframes, using the ``.ak`` accessor.

### Features

- numpy-like API for slicing and accessing data deep in nested structures
- apply any function, ufunc or aggregation at any level
- use with different backends: pandas, polars (lazy and eager),
  dask-dataframe, pyspark, duckDB and cuDF
  exactly the same way (more backends may come in the future)
- CPU and GPU processing and support for ``numba``-jit
- attach object-like behaviours to your record (sub)structures

See the [quick
start](https://akimbo.readthedocs.io/en/latest/quickstart.html)
in the documentation for an introduction to akimbo.

Acknowledgements
----------------

Support for this work was provided by NSF grant [OAC-2103945](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2103945).

Work on this repository is supported in part by:

"Anaconda, Inc. - Advancing AI through open source."

.. raw:: html

    <a href="https://anaconda.com/"><img src="https://camo.githubusercontent.com/b8555ef2222598ed37ce38ac86955febbd25de7619931bb7dd3c58432181d3b6/68747470733a2f2f626565776172652e6f72672f636f6d6d756e6974792f6d656d626572732f616e61636f6e64612f616e61636f6e64612d6c617267652e706e67" alt="anaconda logo" width="40%"/></a>
