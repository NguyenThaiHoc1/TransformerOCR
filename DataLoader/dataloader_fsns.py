import tensorflow as tf
from tqdm import tqdm


class Dataset(object):
    def __init__(self, record_path):
        self.record_path = record_path
        zero = tf.zeros([1], dtype=tf.int64)
        self.keys_to_features = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'image/format': tf.io.FixedLenFeature([], tf.string, default_value='png'),
            'image/width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            'image/orig_width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            'image/class': tf.io.FixedLenFeature([37], tf.int64),
            'image/unpadded_class': tf.io.VarLenFeature(tf.int64),
            'image/text': tf.io.FixedLenFeature([1], tf.string, default_value=''),
        }

        self.dataset = None
        self.iterator = None
        self.batch_size = None

    def parse_tfrecord(self, example):
        res = tf.io.parse_single_example(example, self.keys_to_features)
        image = tf.cast(tf.io.decode_jpeg(res['image/encoded'], 3), tf.float32) / 255.0
        label = tf.cast(res['image/class'], tf.float32)
        return image, label

    def load_tfrecord(self, repeat, batch_size, buffer_size=500):
        dataset = tf.data.TFRecordDataset(self.record_path)
        dataset = dataset.map(self.parse_tfrecord)
        # dataset = dataset.shuffle(buffer_size=buffer_size, seed=43)
        if repeat:
            dataset = dataset.repeat()
        self.dataset = dataset.batch(batch_size)
        self.batch_size = batch_size
        self.iterator = iter(self.dataset)

    def next_batch(self):
        return self.iterator.get_next()

    def __len__(self):
        assert self.dataset is not None, "Please load record after get leng of data"
        count = 0
        for _ in tqdm(self.dataset):
            count += 1
        return count * self.batch_size
