akimbo
==============

.. currentmodule:: akimbo

Top Level Functions
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   read_parquet
   read_json

Accessor
~~~~~~~~

.. currentmodule:: akimbo.mixin

.. autosummary::
   :toctree: generated/

   akimbo.mixin.Accessor


Backends
~~~~~~~~

.. autoclass:: akimbo.pandas.PandasAwkwardAccessor

.. autoclass:: akimbo.dask.DaskAwkwardAccessor

.. autoclass:: akimbo.polars.PolarsAwkwardAccessor


Extensions
~~~~~~~~~~

The following properties appear on the ``.ak`` accessor for data-type
specific functions, mapped onto the structure of the column/frame
being acted on. Check the ``dir()`` of each (or use tab-completion)
to see the operations available.

.. autoclass:: akimbo.datetimes.DatetimeAccessor
   :members: cast

.. autoclass:: akimbo.strings.StringAccessor
   :members:
