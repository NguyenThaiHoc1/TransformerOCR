import tensorflow as tf


class TFSampleArchiscribeCorpus(object):
    """
    This TFSample that using for face recognition
    """

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

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
    def create(img_encoded, width, height, token_ids, text):
        feature = {
            'image/img_encoded': TFSampleArchiscribeCorpus._bytes_feature(img_encoded),
            'image/width': TFSampleArchiscribeCorpus._int64_feature([width]),
            'image/height': TFSampleArchiscribeCorpus._int64_feature([height]),
            'image/token_ids': TFSampleArchiscribeCorpus._int64_feature(token_ids),
            'image/text': TFSampleArchiscribeCorpus._bytes_feature(text)
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))
