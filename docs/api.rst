akimbo
==============

Accessor
~~~~~~~~

.. currentmodule:: akimbo.mixin

.. autosummary::
   :toctree: generated/

   Accessor

.. autoclass:: Accessor
   :members:


Backends
~~~~~~~~

.. autosummary::
    akimbo.pandas.PandasAwkwardAccessor
    akimbo.dask.DaskAwkwardAccessor
    akimbo.polars.PolarsAwkwardAccessor
    akimbo.cudf.CudfAwkwardAccessor
    akimbo.ray.RayAccessor
    akimbo.spark.SparkAccessor

.. autoclass:: akimbo.pandas.PandasAwkwardAccessor

.. autoclass:: akimbo.dask.DaskAwkwardAccessor

.. autoclass:: akimbo.polars.PolarsAwkwardAccessor

.. autoclass:: akimbo.cudf.CudfAwkwardAccessor

.. autoclass:: akimbo.ray.RayAccessor

.. autoclass:: akimbo.spark.SparkAccessor

Top Level Functions
~~~~~~~~~~~~~~~~~~~
.. currentmodule:: akimbo


.. autosummary::
   :toctree: generated/

   read_parquet
   read_json
   read_avro
   get_parquet_schema
   get_json_schema
   get_avro_schema


Extensions
~~~~~~~~~~

The following properties appear on the ``.ak`` accessor for data-type
specific functions, mapped onto the structure of the column/frame
being acted on. Check the ``dir()`` of each (or use tab-completion)
to see the operations available.

.. autoclass:: akimbo.datetimes.DatetimeAccessor
   :members:

.. autoclass:: akimbo.strings.StringAccessor
   :members:

.. raw:: html

    <script data-goatcounter="https://akimbo.goatcounter.com/count"
            async src="//gc.zgo.at/count.js"></script>

The cuDF backend also has these implemented with GPU-specific variants,
``akimbo.cudf.CudfStringAccessor`` and ``akimbo.cudf.CudfDatetimeAccessor``.
