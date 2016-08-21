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


def lcm(a,b):
    import fractions
    return abs(a * b) // fractions.gcd(a, b) if a and b else 0


class TestTableRow(tables.IsDescription):
    col_A = tables.UInt32Col(shape=test_table_col_A_shape)
    col_B = tables.Float64Col(shape=test_table_col_B_shape)


def get_batches(array, size):
    return [ array[i:i+size] for i in range(0, len(array), size)]


def assert_array_equal(self, a, b):
    self.assertTrue(np.all(a == b),
                    msg="LHS: \n" + str(a) + "\n RHS: \n" + str(b))


def assert_items_equal(self, a, b, key, epsilon=0):
    a = [item for sublist in a for item in sublist]
    b = [item for sublist in b for item in sublist]
    self.assertEqual(len(a), len(b))
    a_sorted, b_sorted = sorted(a, key=key), sorted(b, key=key)

    unique_a, counts_a = np.unique(a, return_counts=True)
    unique_b, counts_b = np.unique(b, return_counts=True)

    assert_array_equal(self, unique_a, unique_b)

    epsilon *= np.prod(a[0].shape)
    delta = counts_a - counts_b
    non_zero = np.abs(delta) > 0
    n_non_zero = np.sum(non_zero)
    self.assertLessEqual(n_non_zero, epsilon, msg="Num. zero deltas=" + str(n_non_zero) + " epsilon=" + str(epsilon)
                                                  + "\n" + str(np.unique(delta, return_counts=True))
                                                  + "\n" + str(delta))


class BufferTest(tf.test.TestCase):

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

        test_file.close()

    def tearDown(self):
        import time
        time.sleep(5)
        shutil.rmtree(self.test_dir)

    def test_reader(self):
        N = 4
        N_threads = 4

        def set_up(path, array, batchsize, get_tensors):
            blocksize = batchsize*2 + 1
            reader = tftables.open_file(self.test_filename, batchsize)
            cycles = lcm(len(array), blocksize)//len(array)
            batch = reader.get_batch(path, block_size=blocksize)
            batches = get_batches(array, batchsize)*cycles*N_threads
            loader = tftables.FIFOQueueLoader(reader, N, get_tensors(batch), threads=N_threads)
            return reader, loader, batches

        array_batchsize = 10
        array_reader, array_loader, array_batches = set_up(self.test_array_path, self.test_array,
                                                           array_batchsize, lambda x: [x])
        array_data = array_loader.dequeue()
        array_result = []

        table_batchsize = 5
        table_reader, table_loader, table_batches = set_up(self.test_table_path, self.test_table_ary,
                                                           table_batchsize, lambda x: [x['col_A'], x['col_B']])
        table_A_data, table_B_data = table_loader.dequeue()
        table_result = []

        with self.test_session() as sess:
            sess.run(tf.initialize_all_variables())

            array_loader.start(sess)
            table_loader.start(sess)

            for i in tqdm.tqdm(range(len(array_batches))):
                array_result.append(sess.run(array_data))
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

    def test_cyclic_option(self):
        reader = tftables.open_file(self.test_filename, 10)
        with self.assertRaises(ValueError):
            batch = reader.get_batch("", cyclic=False)
        reader.close()

    def test_uint64(self):
        reader = tftables.open_file(self.test_filename, 10)
        with self.assertRaises(ValueError):
            batch = reader.get_batch("/test_uint64")
        reader.close()

    def test_quick_start_A(self):
        my_network = lambda x: x
        N = 100

        # Open the HDF5 file. The batch_size defined the length
        # (in the outer dimension) of the elements (batches) returned
        # by the reader.
        reader = tftables.open_file(filename=self.test_filename,
                                    batch_size=20)

        # For simple arrays, the get_batch method returns a
        # placeholder for one batch taken from the array.
        array_batch_placeholder = reader.get_batch(self.test_array_path)
        # We can then do a transform on the raw data.
        array_float = tf.to_float(array_batch_placeholder)

        # The placeholder can then be used in your network
        result = my_network(array_float)

        with tf.Session() as sess:
            # The feed method provides a generator that returns
            # feed_dict's containing batches from your HDF5 file.
            for i, feed_dict in enumerate(reader.feed()):
                sess.run(result, feed_dict=feed_dict)
                if i >= N:
                    break

        # Finally, the reader should be closed.
        reader.close()

    def test_quick_start_B(self):
        my_network = lambda x: x
        N = 100

        reader = tftables.open_file(filename=self.test_filename,
                                    batch_size=20)

        # For tables and compound data types, a dictionary is returned.
        table_batch_dict = reader.get_batch(self.test_table_path)
        # The keys for the dictionary are taken from the column names of the table.
        # The values of the dictionary are the corresponding placeholders for the batch.
        col_A_pl, col_B_pl = table_batch_dict['col_A'], table_batch_dict['col_B']

        # You can access multiple datasets within the HDF5 file.
        # They all share the same batchsize, and are fed into your
        # graph simultaneously.
        labels_batch = reader.get_batch(self.test_array_path)
        truth_batch = tf.to_float(labels_batch)

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


if __name__ == '__main__':
    tf.test.main()
