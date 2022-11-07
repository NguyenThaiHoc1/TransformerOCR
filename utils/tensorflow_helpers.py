import tensorflow as tf


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


class TFSample(object):
    """
    This TFSample that using for face recognition
    """

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        # step 1: convert to numpy
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def create(np_image, id_name, filename, img_path, real_name):
        feature = {
            'image/id_name': TFSample._int64_feature(id_name),
            'image/np_image': TFSample._bytes_feature(np_image),
            'image/filename': TFSample._bytes_feature(filename),
            'image/img_path': TFSample._bytes_feature(img_path),
            'image/real_name': TFSample._bytes_feature(real_name)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
