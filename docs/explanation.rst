How it works
============

All of the dataframe libraries we cater to use arrow as internal data storage
(at least, optionally). `Apache Arrow`_ is designed for columnar in-memory
representation of tabular data (i.e., dataframes), but has much additional
functionality such as interoperability with the parquet storage format.

.. _Apache Arrow: https://arrow.apache.org/docs/index.html

While arrow itself and most dataframe-oriented libraries consider simple columns
with types like "numer" and "text" - SQL-like - arrow can store nested records and
variable-length lists while keeping compact storage in arrays of values. Since
the values are in contiguous arrays, data processing can be done with
high-performance vectorised operations.

`Awkward Array`_ is a library for vectorised processing of data containing
nested and variable-length structures, with an API familiar to users
of ``numpy``. It can interoperate with arrow in-memory data.

.. _Awkward Array: https://awkward-array.org/doc/main/

So the purpose of this library, is to bring the efficient processing of
awkward to complex data types in dataframes. Since arrow is a common
denominator for many analysis platforms, we are able to offer the same
API and workflows across all of them.

Where there is no accessor registration mechanism, ``akimbo`` uses
"patching" to attach the ``.ak`` accessor to the target dataframe
objects. This always happens by importing the specific submodule for
the given dataframe library, so others will not be imported/affected
by ``akimbo``.

Lazy versus Eager
-----------------

The dataframe implementations fall into two categories. The "eager" ones
create new dataframes on every operation as soon as the operation is
encountered. For example ``df["a"] + 1`` for Pandas results in new values
being stored in memory immediately. For other libraries, each operation is instead
stored in a pipeline, and only executed when the user has reached the output
they want to see. For instance, ``pyspark`` would not even allow the above
syntax, and ``df.select(col("a") + 1)`` would not cause any work to be
done by itself. This second type, "lazy" dataframes, allow for more optimization,
since the whole workflow is known by the tie compute is requested.

From ``akimbo``'s point of view, this means that for lazy dataframes, we need
to infer the data type that a given operation will produce before seeing any
of the data. This is possible _in_most_cases_, because the data types
are strongly constrained. However, if writing opaque functions (such as
with Numba), you may wish to catch the case where the input is empty,
and explicitly return data of the shape you expect to get with real data. In
all, this should allow the dataframe library to successfully perform
all the optimizations it can; but it's always good practice to select only
the columns and data partitions you expect to need anyway.

Data Round-trips
----------------

Whilst the awkward structure of data is compatible with arrow's, the metadata
is a little different and there are some edge cases. This means, that for every
``ak`` operation, there is some overhead for converting to and from. That means,
that ``akimbo`` will always be a bit slower at performing tasks that your
dataframe library can already do, and this will be more noticeable for many
operations on small frames as opposed to few operations on big frames.

It might be beneficial in such cases to use ``.ak.apply`` with or
without Numba, and do many awkward operations in one go per round-trip.
