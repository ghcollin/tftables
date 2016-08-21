import tensorflow as tf
import multitables as mtb
import numpy as np
import threading


def open_file(filename, batch_size, **kw_args):
    """
    Open a HDF5 file for streaming with multitables.
    Batches will be retrieved with size batch_size.
    Additional keyword arguments will be passed to the multitables.Streamer object.

    :param filename: Filename for the HDF5 file to be read.
    :param batch_size: The size of the batches to be fetched by this reader.
    :param kw_args: Optional arguments to pass to multitables.
    :return: A TableReader instance.
    """
    return TableReader(filename, batch_size, **kw_args)


class TableReader:
    def __init__(self, filename, batch_size, **kw_args):
        """
        Create a HDF5 file reader that reads batches of size batch_size.

        :param filename: The HDF5 file to read.
        :param batch_size: The size of the batches to be read.
        :param kw_args: Optional arguments to pass to multitables.
        """
        self.streamer = mtb.Streamer(filename, **kw_args)
        self.vars = []
        self.batch_size = batch_size
        self.queues = []

    @staticmethod
    def __match_slices(slice1, len1, slice2):
        """
        Assures that the two given slices are compatible with each other and slice1 does no extend past the end
        of an array with length len1.
        If slice1 would extend greater than len1, then slice1 is spliced to wrap around len1.
        slice2 would then be spliced to match the two new slices for slice1.

        :param slice1: Slice that will be checked against slice1.
        :param len1: The length of an array that slice1 should wrap around.
        :param slice2: The slice that should be spliced to match slice1.
        :return: Two tuples.
            Tuple 1 contains two slices that correspond to the non-wraped part of slice1 and slice2.
            Tuple 2 contains two slices that correspond to the wrapped part of slice1 and slice2.
        """
        delta_A, delta_B = len1 - slice1.start, slice1.stop - len1

        slice1_A = slice(slice1.start, slice1.start + delta_A)
        slice2_A = slice(slice2.start, slice2.start + delta_A)

        slice1_B = slice(0, 0 + delta_B)
        slice2_B = slice(slice2_A.stop, slice2_A.stop + delta_B)
        return (slice1_A, slice2_A), (slice1_B, slice2_B)

    @staticmethod
    def __to_tf_dtype(np_dtype):
        """
        Converts a numpy dtype to a tensorflow dtype.
        This may return a larger dtype if no exact fit to np_dtype can be made.

        :param np_dtype: The numpy dtype to convert
        :return: A tensorflow dtype that matches np_dtype as closely as possible.
        """
        # We try converting first so that the code gracefully falls back if tensorflow one day supports uint32/64.
        try:
            return tf.as_dtype(np_dtype)
        except TypeError as e:
            # there is no tensorflow dtype for uint32 at the moment, but we can stuff these into int64s safely
            if np_dtype == np.uint32:
                return tf.int64
            elif np_dtype == np.uint64:
                raise ValueError("Arrays with 64-bit unsigned integer type are not supported, as Tensorflow "
                                 + "has no corresponding data type.")
            raise e

    @staticmethod
    def __create_placeholders(type, batch_shape):
        """
        Recursive function for creating placeholders. If the type is simple (not-compound) then a single tensorflow
        placeholder is returned will the appropriate batch_shape.
        If the type is compound, then a dictionary is returned. Each key of the dictionary corresponds to a
        column (or element) of the compound type. Each value of the dictionary contains the corresponding placeholder.

        The placeholders for this dictionary are created by recursively calling this function. Thus, a tree of
        dictionaries is created if the compound type contains other compound types.

        :param type: The corresponding numpy data type for this placeholder.
        :param batch_shape: The shape of the batch for this placeholder.
        :return: Either a placeholder, or a dictionary of placeholders.
        """
        # If .fields is None, then the array is just a simple (not-compound) array. So a single placeholder is returned.
        if type.fields is None:
            placeholder = tf.placeholder(shape=batch_shape, dtype=TableReader.__to_tf_dtype(type))
            result = placeholder

        # Otherwise, a dictionary of placeholders is needed:
        # As tensorflow doesn't support tensors with compound (or 'structured') types, a tensor (and thus placeholder)
        # if needed for each column in this array.
        else:
            placeholders = {}
            for name in type.fields.keys():
                field_dtype = type.fields[name][0]  # np dtype for the column
                subdtype = field_dtype.subdtype

                # The subdtype will be None, if this is a scalar.
                if subdtype is None:
                    placeholder = TableReader.__create_placeholders(field_dtype, batch_shape)
                    placeholders[name] = placeholder
                # If the column contains a sub-array, then subdtype is not None.
                else:
                    subfield_type, subfield_shape = subdtype  # subfield_shape is a shape of the sub-array
                    # Append the sub-array shape to the batch_shape, as we are creating a single tensor for each column.
                    subfield_batch_shape = batch_shape + list(subfield_shape)
                    placeholder = TableReader.__create_placeholders(subfield_type, subfield_batch_shape)
                    placeholders[name] = placeholder
            result = placeholders

        return result

    def get_batch(self, path, **kw_args):
        """
        Get a Tensorflow placeholder for a batch that will be read from the dataset located at path.
        Additional key word arguments will be forwarded to the get_queue method in multitables.
        Note that passing the 'cyclic' key word argument is forbidden, as this library does not support
        non-cyclic access.

        If the dataset is a table (or other compound-type array) then a dictionary of placeholders will be returned
        instead. The keys of this dictionary correspond to the column names of the table (or compound sub-types).

        :param path: The internal HDF5 path to the dataset to be read.
        :param kw_args: Optional arguments to be forwarded to multitables.
        :return: Either a placeholder or a dictionary depending on the type of dataset.
            If the dataset is a plain array, a placeholder representing once batch is returned.
            If the dataset is a table or compound type, a dictionary of placeholders is returned.
        """
        if 'cyclic' in kw_args:
            raise ValueError("This library does not support non-cyclic access.")
        queue = self.streamer.get_queue(path=path, cyclic=True, **kw_args)
        block_size = queue.block_size
        # get an example for finding data types and row sizes.
        example = self.streamer.get_remainder(path, block_size)
        batch_type = example.dtype
        inner_shape = example.shape[1:]
        batch_shape = [self.batch_size] + list(inner_shape)

        # Generator for reading batches.
        def read_batch():
            # A 'scratch' space of one batch is needed to take care of remainder elements.
            # Here, remainder elements are defined as those left over when the batch size does not divide
            # the block size evenly.
            batch_offset = 0
            batch = np.zeros(batch_shape, dtype=batch_type)

            while True:
                with queue.get() as block:
                    # First, if the batch size is smaller than the block size, then
                    # batches are extracted from the block as yielded.
                    indexes = range(0, block_size, self.batch_size)
                    for start, end in zip(indexes[:-1], indexes[1:]):
                        yield block[start:end]

                    # However, if the batch size is larger than the block size, or the
                    # batch size does not divide the block size evenly, then there will be remainder elements.
                    remainder = slice(indexes[-1], block_size)
                    # These remainder elements will be written into the scratch batch, starting at the current offset.
                    write_slice = slice(batch_offset, batch_offset + (remainder.stop - remainder.start))

                    if write_slice.stop < self.batch_size:
                        batch[write_slice] = block[remainder]
                    # It is possible though, that the remainder elements will write off the end of the scratch block.
                    else:
                        # In this case, the remainder elements need to be split into 2 groups: Those
                        # before the end (slices_A) and those after (slices_B). slices_B will then wrap
                        # around to the start of the scratch batch.
                        slices_A, slices_B = TableReader.__match_slices(write_slice, self.batch_size, remainder)
                        # Write the before group.
                        batch[slices_A[0]] = block[slices_A[1]]
                        # The scratch batch is now full, so yield it.
                        yield batch
                        # Now that the batch was yieled, it is safe to write to the front of it.
                        batch[slices_B[0]] = block[slices_B[1]]
                        # Reset the write_slice so that batch_offset will be updated correctly.
                        write_slice = slices_B[0]

                    # Update the batch_offset, now the remainder elements are written.
                    batch_offset = write_slice.stop

        result = TableReader.__create_placeholders(batch_type, batch_shape)

        self.vars.append((read_batch, result))
        self.queues.append(queue)

        return result

    @staticmethod
    def __feed_batch(feed_dict, batch, placeholders):
        """
        Recursive function for filling in the feed_dict. This recursively walks the dictionary tree given
        by placeholders and adds an element to feed_dict for each leaf.

        :param feed_dict: The feed_dict to fill.
        :param batch: The batch containing the data to be fed.
        :param placeholders: Either a single placeholder, or a dictionary of placeholders.
        :return: None
        """
        if isinstance(placeholders, dict):
            for name in placeholders.keys():
                TableReader.__feed_batch(feed_dict, batch[name], placeholders[name])
        else:
            feed_dict[placeholders] = batch

    def feed(self):
        """
        Generator for feeding a tensorflow operation. Each iteration returns a feed_dict that contains
        the data for one batch. This method reads data for *all* placeholders created.

        :return: A generator which yields tensorflow feed_dicts
        """
        # The reader generator is initialised here to allow safe multi-threaded access to the reader.
        generators = [(reader(), placeholders) for reader, placeholders in self.vars]
        while True:
            feed_dict = {}
            for gen, placeholders in generators:
                # Get the next batch
                batch = next(gen)
                # Populate the feed_dict with the elements of this batch.
                TableReader.__feed_batch(feed_dict, batch, placeholders)
            yield feed_dict

    def close(self):
        """
        Closes the internal queue, signaling the background processes to stop.
        This calls the multitables.Streamer.Queue.close method.

        :return: None
        """
        for q in self.queues:
            q.close()


class FIFOQueueLoader:
    def __init__(self, reader, size, inputs, threads=1):
        """
        Creates a loader that populates a Tensorflow FIFOQueue.
        Experimentation suggests this tends to perform best when threads=1.
        The graph defined by the inputs should only contain placeholders created by the supplied reader object.

        :param reader: An instance of the associated TableReader class.
        :param size: The max size of the iternal queue.
        :param inputs: A list of tensors that will be stored in the queue.
        :param threads: Number of background threads to populate the queue with.
        """
        self.reader = reader
        self.coord = tf.train.Coordinator()
        self.q = tf.FIFOQueue(size, [i.dtype for i in inputs], [i.get_shape() for i in inputs])
        self.enq_op = self.q.enqueue(inputs)
        self.q_close_op = self.q.close(cancel_pending_enqueues=True)
        self.n_threads = threads
        self.threads = []

    def __read_thread(self, sess):
        """
        Function that defines the background threads. Feeds data from the reader into the FIFOQueue.

        :param sess: Tensorflow session.
        :return:
        """
        with self.coord.stop_on_exception():
            for feed_dict in self.reader.feed():
                sess.run(self.enq_op, feed_dict=feed_dict)

                if self.coord.should_stop():
                    break

    def dequeue(self):
        """
        Returns a dequeue operation. Elements defined by the input tensors and supplied by the reader
        are returned from this operation. This calls the dequeue method on the internal Tensorflow FIFOQueue.

        :return: A dequeue operation.
        """
        return self.q.dequeue()

    def start(self, sess):
        """
        Starts the background threads. The enqueue operations are run in the given Tensorflow session.

        :param sess: Tensorflow session.
        :return: None
        """
        for _ in range(self.n_threads):
            t = threading.Thread(target=FIFOQueueLoader.__read_thread, args=(self, sess))
            t.daemon = True
            t.start()
            self.threads.append(t)

    def stop(self, sess):
        """
        Stops the background threads, and joins them. This should be called after all operations are complete.

        :param sess: The Tensorflow operation that this queue loader was started with.
        :return:
        """
        self.coord.request_stop()
        sess.run(self.q_close_op)
        self.coord.join(self.threads)
