import tensorflow as tf
import numpy as np
from extract_modis_data import *
from autoencoder import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(patch):
    """Serializes one data example."""
    # Store image depth
    depth = patch.shape[-1]

    feature = {
        'depth': _int64_feature(depth),
        'patch': _bytes_feature(patch),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

# Writing to TFRecord
def write_tfrecord(filename, dataset):
    with tf.io.TFRecordWriter(filename) as writer:
        for patch in dataset:
            serialized_example = serialize_example(patch)
            writer.write(serialized_example)


