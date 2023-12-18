Installation
============

Requirements
~~~~~~~~~~~~

To install ``awkward-pandas`` you will need both ``awkward`` and
``pandas``. Whether you install from PyPI or conda-forge, these
requirements will be enforced. The strict requirements are:

- ``awkward >=2.0.0``
- ``pandas >=1.2``

From PyPI
~~~~~~~~~

.. code-block:: none

   $ pip install awkward-pandas

From conda-forge
~~~~~~~~~~~~~~~~

.. code-block:: none

   $ conda install awkward-pandas -c conda-forge

Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

- ``pyarrow``: for reading and writing data stored in parquet files.
- ``s3fs``: for reading data from or writing data to Amazon S3.
