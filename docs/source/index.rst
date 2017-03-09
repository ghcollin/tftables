.. tftables documentation master file, created by
   sphinx-quickstart on Tue Mar  7 21:24:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

tftables documentation
**********************

`tftables <https://github.com/ghcollin/tftables>`_ allows convenient access to HDF5 files with Tensorflow.
A class for reading batches of data out of arrays or tables is provided.
A secondary class wraps both the primary reader and a Tensorflow FIFOQueue for straight-forward streaming
of data from HDF5 files into Tensorflow operations.

The library is backed by `multitables <https://github.com/ghcollin/multitables>`_ for high-speed reading of HDF5
datasets. ``multitables`` is based on PyTables (``tables``), so this library can make use of any compression algorithms
that PyTables supports.

Contents
========

.. toctree::
   :maxdepth: 2

   quick
   howto
   reference

Licence
=======

This software is distributed under the MIT licence.
See the `LICENSE.txt <https://github.com/ghcollin/tftables/blob/master/LICENSE.txt>`_ file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

