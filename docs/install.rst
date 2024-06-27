Installation
============

Requirements
~~~~~~~~~~~~

To install ``akimbo`` you will need ``awkward`` and
one of the backend libraries: ``pandas``, ``dask`` or ``polars``.
In addition, for string and datetime functions, you will need ``pyarrow``.


From PyPI
~~~~~~~~~

.. code-block:: none

   $ pip install akimbo

From conda-forge
~~~~~~~~~~~~~~~~

.. code-block:: none

   $ conda install akimbo -c conda-forge

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

- ``pyarrow``: for reading and writing data stored in parquet files, and for
  for the ``.str`` and ``.dt`` accessor attributes.
