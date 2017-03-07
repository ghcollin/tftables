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

    pip install git+https://github.com/ghcollin/tftables.git

or download and run

::

    python setup.py install

or simply copy ``tftables.py`` into your project directory.

``tftables`` depends on ``multitables``, ``numpy`` and ``tensorflow``.

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
        loader = tftables.load_dataset(filename=self.test_filename,
                                       dataset_path=self.test_mock_data_path,
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

A slightly more involved example showing how to access multiple datasets in one HDF5 file,
as well as the full API.

.. code:: python

    import tftables
    import tensorflow as tf
    reader = tftables.open_file(filename='path/to/h5_file',
                                batch_size=20)

    # For tables and compound data types, a dictionary is returned.
    table_batch_dict = reader.get_batch('/internal/h5_path/to/table')
    # The keys for the dictionary are taken from the column names of the table.
    # The values of the dictionary are the corresponding placeholders for the batch.
    col_A_pl, col_B_pl = table_batch_dict['col_A'], table_batch_dict['col_B']

    # You can access multiple datasets within the HDF5 file.
    # They all share the same batchsize, and are fed into your
    # graph simultaneously.
    labels_batch = reader.get_batch('/my_label_array')
    truth_batch = tf.one_hot(labels_batch, 2, 1, 0)

    # This class creates a Tensorflow FIFOQueue and populates it with data from the reader.
    loader = tftables.FIFOQueueLoader(reader, size=2,
    # The inputs are placeholders (or graphs derived thereof) from the reader.
        inputs=[col_A_pl, col_B_pl, truth_batch])
    # Batches are taken out of the queue using a dequeue operation.
    dequeue_op = loader.dequeue()

    # The dequeued data can then be used in your network.
    result = my_network(dequeue_op)

    with tf.Session() as sess:
        # The queue loader needs to be started inside your session
        loader.start(sess)

        # Then simply run your operation, data will be streamed
        # out of the HDF5 file and into your graph!
        for _ in range(N):
            sess.run(result)

        # Finally, the queue should be stopped.
        loader.stop(sess)
    reader.close()


Examples
========

See the `unit tests <https://github.com/ghcollin/tftables/blob/master/tftables_test.py>`_ for complete examples.
