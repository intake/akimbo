
Akimbo
======

The akimbo project provides a Dataframe accessor for various backend, that enable
analysis of nested, non-tabular data in
workflows. This will be much faster and memory efficient than iterating
over python dicts/lists, which quickly becomes unfeasible for big data.

When you import ``kimbo``, a new ``.ak`` accessor will appear on your
dataframes, allowing the fast vectorized processing of "awkward" data
(nested structures and variable-length ragged lists) held in columns.

Features
--------

Multi library support
~~~~~~~~~~~~~~~~~~~~~

Currently, we support the following dataframe libraries with
identical syntax:

- pandas
- dask.dataframe
- polars
- cuDF (in development)


numpy-like API
~~~~~~~~~~~~~~

for slicing and accessing data deep in nested structures,

Example: choose every second inner element in a list-of-lists

.. code-block:: python
    series.ak[:, ::2]

Any function, ufunc or aggregation at any level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For manipulating numerics at deeper levels of your nested structures or
ragged arrays while maintaining the original layout

.. code-block:: python
    series.ak.abs()  # absolute for all numerical values
    series.ak.sum(axis=3)  # sum over deeply nested level
    series.ak + 1  # numpy-like broadcasting into deeper levels

You can even apply string and datetime operations to ragged/nested
arrays of values, and they will only affect the appropriate parts of the structure
without changing the layout.

.. code-block:: python
    series.ak.str.upper()

CPU/GPU numba support
~~~~~~~~~~~~~~~~~~~~~

Pass nested functions to numba for compiled-speed computes over your
data where you need an algorithm more complex than can be easily
written with the numpy-like API. This can also be used for aggregations
in groupby/window operations. If your data is on the GPU, you can
use numba-cuda with slight modifications to your original function.

.. code-block:: python
    @numba.njit
    def sum_list_of_list(x):
        total = 0
        for x0 in x:
            for x1 in x0:
                total += x1
        return total


    series.ak.apply(sum_list_of_lists)

Object Behaviours
~~~~~~~~~~~~~~~~~

Where your struct has higher-level concept associated with it - the
fields have logical relationship with each other - you can define a
class to encode these behaviours as methods. For instance, you can
describe that an array of (x, y, z) is in fact a set of points in
3D space. The methods you
define will appear on the ``.ak`` accessor or can be used for ufunc and
operator overloads.


Sub-accessors
~~~~~~~~~~~~~

As an alternative to the object-oriented behaviours, developers may create
accessor namespaces that appear under ``.ak`` similar to the the builtin
``.ak.str`` (strings ops) snd ``.ak.dt`` (datetime ops) included already.

One experimental proof-of-concept is `akimbo-ip`_, which provides fast vectorised
manipulations of IPv4/6 addresses and networks; and by using this through
the ``akimbo`` system, you can apply these methods to ragged/nested dataframes.

.. _akimbo-ip: https://github.com/intake/akimbo-ip

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   install.rst
   quickstart.ipynb

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api.rst


.. raw:: html

    <script data-goatcounter="https://akimbo.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>
