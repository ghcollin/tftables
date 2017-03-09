How To
******

Use of the library starts with creating a ``TableReader`` object.

.. code:: python

    import tftables
    reader = tftables.open_file(filename="/path/to/h5/file", batch_size=10)

Here the batch size is specified as an argument to the ``open_file`` function. The batch_size defines the length
(in the outer dimension) of the elements (batches) returned by ``reader``.

Accessing a single array
========================

Suppose you only want to read a single array from your HDF5 file. Doing this is quite straight-forward.
Start by getting a tensorflow placeholder for your batch from ``reader``.

.. code:: python

    array_batch_placeholder = reader.get_batch(
        path = '/h5/path',  # This is the path to your array inside the HDF5 file.
        cyclic = True,      # In cyclic access, when the reader gets to the end of the
                            # array, it will wrap back to the beginning and continue.
        ordered = False     # The reader will not require the rows of the array to be
                            # returned in the same order as on disk.
    )

    # You can transform the batch however you like now.
    # For example, casting it to floats.
    array_batch_float = tf.to_float(array_batch_placeholder)

    # The data can now be fed into your network
    result = my_network(array_batch_float)

    with tf.Session() as sess:
        # The feed method provides a generator that returns
        # feed_dict's containing batches from your HDF5 file.
        for i, feed_dict in enumerate(reader.feed()):
            sess.run(result, feed_dict=feed_dict)
            if i >= N:
                break

    # Finally, the reader should be closed.
    reader.close()

Note that be default, the ``ordered`` argument to ``get_batch`` is set to ``True``. If you require the rows of the
array to be returned in the same order as they are on disk, then you should leave it as ``ordered = True``.
However, this may result in a performance penalty. In machine learning, rows of a dataset often represent
independent examples, or data points. Thus their ordering is not important.

Accessing a single table
========================

When reading from a table, the ``get_batch`` method returns a dictionary. The columns of the table form the keys
of this dictionary, and the values are tensorflow placeholders for batches of each column. If one of the columns has
a compound datatype, then its corresponding value in the dictionary will itself be a dictionary. In this way,
recursive compound datatypes will give recursive dictionaries.

For example, if your table just had two columns, named ``label`` and ``data``, then you could use:

.. code:: python

    table_batch = reader.get_batch(
        path = '/path/to/table',
        cyclic = True,
        ordered = False
    )

    label_batch = table_batch['label']
    data_batch = table_batch['data']

If your table was a bit more complicated, with columns named ``label`` and ``value``. And the ``value`` column has
a compound type with fields named ``image`` and ``lidar``, then you could use:

.. code:: python

    table_batch = reader.get_batch(
        path = '/path/to/complex_table',
        cyclic = True,
        ordered = False
    )

    label_batch = table_batch['label']
    value_batch = table_batch['value']

    image_batch = value_batch['image']
    lidar_batch = value_batch['lidar']

Using a FIFO queue
==================

Copying data to the GPU through a ``feed_dict`` is notoriously slow in Tensorflow. It is much faster to buffer
data in a queue. You are free to manage your own queues, but a helper class is included to make this task easier.

.. code:: python

    # As before
    array_batch_placeholder = reader.get_batch(
        path = '/h5/path',
        cyclic = True,
        ordered = False)
    array_batch_float = tf.to_float(array_batch_placeholder)

    # Now we create a FIFO Loader
    loader = reader.get_fifoloader(
        queue_size = 10,              # The maximum number of elements that the
                                      # internal Tensorflow queue should hold.
        inputs = [array_batch_float], # A list of tensors that will be stored
                                      # in the queue.
        threads = 1                   # The number of threads used to stuff the
                                      # queue. If ordered access to a dataset
                                      # was requested, then only 1 thread
                                      # should be used.
    )

    # Batches can now be dequeued from the loader for use in your network.
    array_batch_cpu = loader.dequeue()
    result = my_network(array_batch_cpu)

    with tf.Session() as sess:

        # The loader needs to be started with your Tensorflow session.
        loader.start(sess)

        for i in range(N):
            # You can now cleanly evaluate your network without a feed_dict.
            sess.run(result)

        # It also needs to be stopped for clean shutdown.
        loader.stop(sess)

    # Finally, the reader should be closed.
    reader.close()

Non-cyclic access
-----------------

If you are classifying a dataset, rather than training a model, then you probably only want to run through the
dataset once. This can be done by passing ``cyclic = False`` to ``get_batch``. Once finished, the internal Tensorflow
queue will throw an instance of the ``tensorflow.errors.OutOfRangeError`` exception to signal termination of the loop.

This can be caught manually with a try-catch block:

.. code:: python

    with tf.Session() as sess:
        loader.start(sess)

        try:
            # Keep iterating until the exception breaks the loop
            while True:
                sess.run(result)
        # Now silently catch the exception.
        except tf.errors.OutOfRangeError:
            pass

        loader.stop(sess)

A slightly more elegant solution is to use a context manager supplied by the loader class:

.. code:: python

    with tf.Session() as sess:
        loader.start(sess)

        # This context manager suppresses the exception.
        with loader.catch_termination():
            # Keep iterating until the exception breaks the loop
            while True:
                sess.run(result)

        loader.stop(sess)

Start stop context manager
--------------------------

In either cyclic or non-cyclic access, we can use a context manager to start and stop the loader class.

.. code:: python

    with tf.Session() as sess:
        with loader.begin(sess):
            # Loop

Quick access to a single dataset
================================

It is highly recommended that you use a single dataset, this allows you to use unordered access which is a fastest
way of reading data. If you have multiple sources of data, such as labels and images, then you should organise them
into a table. This also has performance benefits due to the locality of the data.

When you only have one dataset, the function ``load_dataset`` is provided to set up the reader and loader for you.
Any preprocessing that need to be done CPU side before loading into the queue can be written as a function that
generates a Tensorflow graph. This input transformation function is fed into ``load_dataset`` as an argument.

The input transform function should return a list of tensors that will be stored in the queue. The input transform
is required when the dataset is a table, as the dictionary needs to be turned into a list.

.. code:: python

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

When using ``load_dataset`` the reader is automatically closed when the loader is stopped.

Accessing multiple datasets
===========================

If your HDF5 file has multiple datasets (multiple arrays, tables or both) then you should write a script to transform
it into a file with only a single table. If this isn't possible, then you can access the datasets directly through
``tftables``, but must do so using ordered access (otherwise the datasets can get out of sync).

.. code:: python

    # Use get_batch to access the table.
    # Both datasets must be accessed in ordered mode.
    table_batch_dict = reader.get_batch(
        path = '/internal/h5_path/to/table',
        ordered = True)
    col_A_pl, col_B_pl = table_batch_dict['col_A'], table_batch_dict['col_B']

    # Now use get_batch again to access an array.
    # Both datasets must be accessed in ordered mode.
    labels_batch = reader.get_batch('/my_label_array', ordered = True)
    truth_batch = tf.one_hot(labels_batch, 2, 1, 0)

    # The loader takes a list of tensors to be stored in the queue.
    # When accessing in ordered mode, threads should be set to 1.
    loader = reader.get_fifoloader(
        queue_size = 10,
        inputs = [truth_batch, col_A_pl, col_B_pl],
        threads = 1)

    # Batches are taken out of the queue using a dequeue operation.
    # Tensors are returned in the order they were given when creating the loader.
    truth_cpu, col_A_cpu, col_B_cpu = loader.dequeue()

    # The dequeued data can then be used in your network.
    result = my_network(truth_cpu, col_A_cpu, col_B_cpu)

    with tf.Session() as sess:
        with loader.begin(sess):
            for _ in range(N):
                sess.run(result)

    reader.close()

Ordered access is enabled be default when using ``get_batch`` as a safety measure. It is disabled when using
``load_dataset`` as that function restricts access to a single dataset.