akimbo
==============

.. currentmodule:: akimbo

Top Level Functions
~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   read_parquet
   read_json
   read_avro
   get_parquet_schema
   get_json_schema
   get_avro_schema

Accessor
~~~~~~~~

.. currentmodule:: akimbo.mixin

.. autosummary::
   :toctree: generated/

   akimbo.mixin.Accessor

.. autoclass:: akimbo.mixin.Accessor
   :members:


Backends
~~~~~~~~

.. autosummary::
    akimbo.pandas.PandasAwkwardAccessor
    akimbo.dask.DaskAwkwardAccessor
    akimbo.polars.PolarsAwkwardAccessor

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

.. raw:: html

    <script data-goatcounter="https://akimbo.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>
