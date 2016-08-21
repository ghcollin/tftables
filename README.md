This library allows convenient access to HDF5 files with Tensorflow. 
A class for reading batches of data out of arrays or tables is provided.
A secondary class wraps both the primary reader and a Tensorflow FIFOQueue for straight-forward streaming 
of data from HDF5 files into Tensorflow operations.

The libaray is backed by [multitables](https://github.com/ghcollin/multitables) 
for high-speed reading of the HDF5 datasets. 
`multitables` is based on PyTables (`tables`), so this library can make use of any compression algorithms 
that PyTables supports.

# Licence
This software is distributed under the MIT licence. 
See the [LICENSE.txt](https://github.com/ghcollin/tftables/blob/master/LICENSE.txt) file for details.

# Installation
```
pip install git+https://github.com/ghcollin/tftables.git
```
or download and run
```
python setup.py install
```
or simply copy `tftables.py` into your project directory.

`tftables` depends on `multitables`, `numpy` and `tensorflow`.

# Quick start
```python
import tftables
import tensorflow as tf
# Open the HDF5 file. The batch_size defined the length 
# (in the outer dimension) of the elements (batches) returned
# by the reader.
reader = tftables.open_file(filename='path/to/h5_file', 
                            batch_size=20)

# For simple arrays, the get_batch method returns a 
# placeholder for one batch taken from the array.
array_batch_placeholder = reader.get_batch('/internal/h5_path/to/array')
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
```

A slightly more involved example showing how to use HDF5 tables and Tensorflow queues.
```python
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
```

# Examples
See the [unit tests](https://github.com/ghcollin/tftables/blob/master/tftables_test.py) for complete examples.
