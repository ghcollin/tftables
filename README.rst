`tftables <https://github.com/ghcollin/tftables>`_ allows convenient access to HDF5 files with Tensorflow.
A class for reading batches of data out of arrays or tables is provided.
A secondary class wraps both the primary reader and a Tensorflow FIFOQueue for straight-forward streaming 
of data from HDF5 files into Tensorflow operations.

The library is backed by `multitables <https://github.com/ghcollin/multitables>`_ for high-speed reading of HDF5
datasets. ``multitables`` is based on PyTables (``tables``), so this library can make use of any compression algorithms
that PyTables supports.

Licence
=======

This software is distributed under the MIT licence. 
See the `LICENSE.txt <https://github.com/ghcollin/tftables/blob/master/LICENSE.txt>`_ file for details.

Installation
============

::

    pip install tftables

Alternatively, to install from HEAD, run

::

    pip install git+https://github.com/ghcollin/tftables.git

You can also `download <https://github.com/ghcollin/tftables/archive/master.zip>`_
or `clone the repository <https://github.com/ghcollin/tftables>`_ and run

::

    python setup.py install

``tftables`` depends on ``multitables``, ``numpy`` and ``tensorflow``. The package is compatible with the latest versions of python
2 and 3.

Quick start
===========

An example of accessing a table in a HDF5 file.

.. code:: python

    import tftables
    import tensorflow as tf

    with tf.device('/cpu:0'):
        # This function preprocesses the batches before they
        # are loaded into the internal queue.
        # You can cast data, or do one-hot transforms.
        # If the dataset is a table, this function is required.
        def input_transform(tbl_batch):
            labels = tbl_batch['label']
            data = tbl_batch['data']

            truth = tf.to_float(tf.one_hot(labels, num_labels, 1, 0))
            data_float = tf.to_float(data)

            return truth, data_float

        # Open the HDF5 file and create a loader for a dataset.
        # The batch_size defines the length (in the outer dimension)
        # of the elements (batches) returned by the reader.
        # Takes a function as input that pre-processes the data.
        loader = tftables.load_dataset(filename='path/to/h5_file.h5',
                                       dataset_path='/internal/h5/path',
                                       input_transform=input_transform,
                                       batch_size=20)

    # To get the data, we dequeue it from the loader.
    # Tensorflow tensors are returned in the same order as input_transformation
    truth_batch, data_batch = loader.dequeue()

    # The placeholder can then be used in your network
    result = my_network(truth_batch, data_batch)

    with tf.Session() as sess:

        # This context manager starts and stops the internal threads and
        # processes used to read the data from disk and store it in the queue.
        with loader.begin(sess):
            for _ in range(num_iterations):
                sess.run(result)


If the dataset is an array instead of a table. Then ``input_transform`` can be omitted
if no pre-processing is required. If only a single pass through the dataset is desired,
then you should pass ``cyclic=False`` to ``load_dataset``.


Examples
========

See the `unit tests <https://github.com/ghcollin/tftables/blob/master/tftables_test.py>`_ for complete examples.

Examples
========

See the `how-to <http://tftables.readthedocs.io/en/latest/howto.html>`_ for more in-depth documentation, and the
`unit tests <https://github.com/ghcollin/tftables/blob/master/tftables_test.py>`_ for complete examples.

Documentation
=============

`Online documentation <http://tftables.readthedocs.io/en/latest/>`_ is available.
A `how to <http://tftables.readthedocs.io/en/latest/howto.html>`_ gives a basic overview of the library.

Offline documentation can be built from the ``docs`` folder using ``sphinx``.