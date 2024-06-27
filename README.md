# akimbo

**For when your data won't fit in your dataframe**

[![Tests](https://github.com/intake/akimbo/actions/workflows/pypi.yml/badge.svg)](https://github.com/intake/akimbo/actions/workflows/pypi.yml)
[![Documentation Status](https://readthedocs.org/projects/akimbo/badge/?version=latest)](https://akimbo.readthedocs.io/en/latest/?badge=latest)

Akimbo provides fast vectorized processing of nested, ragged data
in dataframes, using the ``.ak`` accessor.

### Features

- numpy-like API for slicing and accessing data deep in nested structures
- apply any function, ufunc or aggregation at any level
- use with different backends: pandas, polars, dask-dataframe and cuDF
  exactly the same way (more backends may come in the future)
- CPU and GPU processing and support for ``numba``-jit
- attach object-like behaviours to your record (sub)structures

See the [quick
start](https://akimbo.readthedocs.io/en/latest/quickstart.html)
in the documentation for an introduction to akimbo.

Acknowledgements
----------------

Support for this work was provided by NSF grant [OAC-2103945](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2103945).
