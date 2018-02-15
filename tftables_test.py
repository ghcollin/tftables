# Copyright (C) 2016 G. H. Collin (ghcollin)
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE.txt file for details.

import tensorflow as tf
import numpy as np
import tables
import tempfile
import os
import shutil
import tqdm

import tftables

test_table_col_A_shape = (100,200)
test_table_col_B_shape = (7,49)


class TestTableRow(tables.IsDescription):
    col_A = tables.UInt32Col(shape=test_table_col_A_shape)
    col_B = tables.Float64Col(shape=test_table_col_B_shape)

test_mock_data_shape = (100, 100)


class TestMockDataRow(tables.IsDescription):
    label = tables.UInt32Col()
    data = tables.Float64Col(shape=test_mock_data_shape)


def lcm(a,b):
    import fractions
    return abs(a * b) // fractions.gcd(a, b) if a and b else 0


def get_batches(array, size, trim_remainder=False):
    result = [ array[i:i+size] for i in range(0, len(array), size)]
    if trim_remainder and len(result[-1]) != len(result[0]):
        result = result[:-1]
    return result


def assert_array_equal(self, a, b):
    self.assertTrue(np.array_equal(a, b),
                    msg="LHS: \n" + str(a) + "\n RHS: \n" + str(b))


def assert_items_equal(self, a, b, key, epsilon=0):
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]
    self.assertEqual(len(a), len(b))
    #a_sorted, b_sorted = (a, b) if key is None else (sorted(a, key=key), sorted(b, key=key))

    unique_a, counts_a = np.unique(a, return_counts=True)
    unique_b, counts_b = np.unique(b, return_counts=True)

    self.assertAllEqual(unique_a, unique_b)

    epsilon *= np.prod(a[0].shape)
    delta = counts_a - counts_b
    self.assertLessEqual(np.max(np.abs(delta)), 1, msg="More than one extra copy of an element.\n" + str(delta)
                                                        + "\n" + str(np.unique(delta, return_counts=True)))
    non_zero = np.abs(delta) > 0
    n_non_zero = np.sum(non_zero)
    self.assertLessEqual(n_non_zero, epsilon, msg="Num. zero deltas=" + str(n_non_zero) + " epsilon=" + str(epsilon)
                                                  + "\n" + str(np.unique(delta, return_counts=True))
                                                  + "\n" + str(delta))


class TFTablesTest(tf.test.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_filename = os.path.join(self.test_dir, 'test.h5')
        test_file = tables.open_file(self.test_filename, 'w')

        self.test_array = np.arange(100*1000).reshape((1000, 10, 10))
        self.test_array_path = '/test_array'
        array = test_file.create_array(test_file.root, self.test_array_path[1:], self.test_array)

        self.test_table_ary = np.array([ (
            np.random.randint(256, size=np.prod(test_table_col_A_shape)).reshape(test_table_col_A_shape),
            np.random.rand(*test_table_col_B_shape)) for _ in range(100) ],
                                       dtype=tables.dtype_from_descr(TestTableRow))
        self.test_table_path = '/test_table'
        table = test_file.create_table(test_file.root, self.test_table_path[1:], TestTableRow)
        table.append(self.test_table_ary)

        self.test_uint64_array = np.arange(10).astype(np.uint64)
        self.test_uint64_array_path = '/test_uint64'
        uint64_array = test_file.create_array(test_file.root, self.test_uint64_array_path[1:], self.test_uint64_array)

        self.test_mock_data_ary = np.array([ (
            np.random.rand(*test_mock_data_shape),
            np.random.randint(10, size=1)[0] ) for _ in range(1000) ],
                                       dtype=tables.dtype_from_descr(TestMockDataRow))
        self.test_mock_data_path = '/mock_data'
        mock = test_file.create_table(test_file.root, self.test_mock_data_path[1:], TestMockDataRow)
        mock.append(self.test_mock_data_ary)

        test_file.close()

    def tearDown(self):
        import time
        time.sleep(5)
        shutil.rmtree(self.test_dir)

    def test_cyclic_unordered(self):
        N = 4
        N_threads = 4

        def set_up(path, array, batchsize, get_tensors):
            blocksize = batchsize*2 + 1
            reader = tftables.open_file(self.test_filename, batchsize)
            cycles = lcm(len(array), blocksize)//len(array)
            batch = reader.get_batch(path, block_size=blocksize, ordered=False)
            batches = get_batches(array, batchsize)*cycles*N_threads
            loader = reader.get_fifoloader(N, get_tensors(batch), threads=N_threads)
            return reader, loader, batches, batch

        array_batchsize = 10
        array_reader, array_loader, array_batches, array_batch_pl = set_up(self.test_array_path, self.test_array,
                                                           array_batchsize, lambda x: [x])
        array_data = array_loader.dequeue()
        array_result = []

        table_batchsize = 5
        table_reader, table_loader, table_batches, table_batch_pl = set_up(self.test_table_path, self.test_table_ary,
                                                           table_batchsize, lambda x: [x['col_A'], x['col_B']])
        table_A_data, table_B_data = table_loader.dequeue()
        table_result = []

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            array_loader.start(sess)
            table_loader.start(sess)

            for i in tqdm.tqdm(range(len(array_batches))):
                array_result.append(sess.run(array_data).copy())
                self.assertEqual(len(array_result[-1]), array_batchsize)

            assert_items_equal(self, array_batches, array_result,
                               key=lambda x: x[0, 0], epsilon=2*N_threads*array_batchsize)

            for i in tqdm.tqdm(range(len(table_batches))):
                result = np.zeros_like(table_batches[0])
                result['col_A'], result['col_B'] = sess.run([table_A_data, table_B_data])
                table_result.append(result)
                self.assertEqual(len(table_result[-1]), table_batchsize)

            assert_items_equal(self, table_batches, table_result,
                               key=lambda x: x[1][0, 0], epsilon=2*N_threads*table_batchsize)

            try:
                array_loader.stop(sess)
                table_loader.stop(sess)
            except tf.errors.CancelledError:
                pass

        array_reader.close()
        table_reader.close()

    def test_shared_reader(self):
        batch_size = 8
        reader = tftables.open_file(self.test_filename, batch_size)

        array_batch = reader.get_batch(self.test_array_path, cyclic=False)
        table_batch = reader.get_batch(self.test_table_path, cyclic=False)

        array_batches = get_batches(self.test_array, batch_size, trim_remainder=True)
        table_batches = get_batches(self.test_table_ary, batch_size, trim_remainder=True)
        total_batches = min(len(array_batches), len(table_batches))

        loader = reader.get_fifoloader(10, [array_batch, table_batch['col_A'], table_batch['col_B']], threads=4)

        deq = loader.dequeue()
        array_result = []
        table_result = []

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            loader.start(sess)

            with loader.catch_termination():
                while True:
                    tbl = np.zeros_like(self.test_table_ary[:batch_size])
                    ary, tbl['col_A'], tbl['col_B'] = sess.run(deq)
                    array_result.append(ary)
                    table_result.append(tbl)


            assert_items_equal(self, array_result, array_batches[:total_batches],
                               key=None, epsilon=0)

            assert_items_equal(self, table_result, table_batches[:total_batches],
                               key=None, epsilon=0)

            loader.stop(sess)

        reader.close()

    def test_uint64(self):
        reader = tftables.open_file(self.test_filename, 10)
        with self.assertRaises(ValueError):
            batch = reader.get_batch("/test_uint64")
        reader.close()


    def test_quick_start_A(self):
        my_network = lambda x, y: x
        num_iterations = 100
        num_labels = 10

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


    def test_howto(self):
        def my_network(*args):
            return args[0]
        N = 100

        reader = tftables.open_file(filename=self.test_filename, batch_size=10)

        # Accessing a single array
        # ========================

        array_batch_placeholder = reader.get_batch(
            path=self.test_array_path,  # This is the path to your array inside the HDF5 file.
            cyclic=True,  # In cyclic access, when the reader gets to the end of the
            # array, it will wrap back to the beginning and continue.
            ordered=False  # The reader will not require the rows of the array to be
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
        #reader.close()

        # Accessing a single table
        # ========================

        table_batch = reader.get_batch(
            path=self.test_mock_data_path,
            cyclic=True,
            ordered=False
        )

        label_batch = table_batch['label']
        data_batch = table_batch['data']

        # Using a FIFO queue
        # ==================

        # As before
        array_batch_placeholder = reader.get_batch(
            path=self.test_array_path,
            cyclic=True,
            ordered=False)
        array_batch_float = tf.to_float(array_batch_placeholder)

        # Now we create a FIFO Loader
        loader = reader.get_fifoloader(
            queue_size=10,  # The maximum number of elements that the
            # internal Tensorflow queue should hold.
            inputs=[array_batch_float],  # A list of tensors that will be stored
            # in the queue.
            threads=1  # The number of threads used to stuff the
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
        #reader.close()

        # Accessing multiple datasets
        # ===========================

        # Use get_batch to access the table.
        # Both datasets must be accessed in ordered mode.
        table_batch_dict = reader.get_batch(
            path=self.test_table_path,
            ordered=True)
        col_A_pl, col_B_pl = table_batch_dict['col_A'], table_batch_dict['col_B']

        # Now use get_batch again to access an array.
        # Both datasets must be accessed in ordered mode.
        labels_batch = reader.get_batch(self.test_array_path, ordered=True)
        truth_batch = tf.one_hot(labels_batch, 2, 1, 0)

        # The loader takes a list of tensors to be stored in the queue.
        # When accessing in ordered mode, threads should be set to 1.
        loader = reader.get_fifoloader(
            queue_size=10,
            inputs=[truth_batch, col_A_pl, col_B_pl],
            threads=1)

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

    def test_howto_quick(self):
        my_network = lambda x, y: x
        num_iterations = 100
        num_labels = 256

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

    def test_howto_cyclic1(self):

        def my_network(*args):
            return args[0]

        reader = tftables.open_file(filename=self.test_filename, batch_size=10)

        # Non-cyclic access
        # -----------------

        array_batch_placeholder = reader.get_batch(
            path=self.test_array_path,
            cyclic=False,
            ordered=False)
        array_batch_float = tf.to_float(array_batch_placeholder)

        loader = reader.get_fifoloader(
            queue_size=10,
            inputs=[array_batch_float],
            threads=1
        )

        array_batch_cpu = loader.dequeue()
        result = my_network(array_batch_cpu)

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

    def test_howto_cyclic2(self):

        def my_network(*args):
            return args[0]

        reader = tftables.open_file(filename=self.test_filename, batch_size=10)

        # Non-cyclic access
        # -----------------

        array_batch_placeholder = reader.get_batch(
            path=self.test_array_path,
            cyclic=False,
            ordered=False)
        array_batch_float = tf.to_float(array_batch_placeholder)

        loader = reader.get_fifoloader(
            queue_size=10,
            inputs=[array_batch_float],
            threads=1
        )

        array_batch_cpu = loader.dequeue()
        result = my_network(array_batch_cpu)

        with tf.Session() as sess:
            loader.start(sess)

            # This context manager suppresses the exception.
            with loader.catch_termination():
                # Keep iterating until the exception breaks the loop
                while True:
                    sess.run(result)

            loader.stop(sess)

if __name__ == '__main__':
    tf.test.main()
