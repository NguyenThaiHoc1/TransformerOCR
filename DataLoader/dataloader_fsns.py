import tensorflow as tf
from tqdm import tqdm
import pdb
import numpy as np

tf.data.experimental.enable_debug_mode()


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

        self.max_ratio = 6

    @tf.function
    def _process_image_ratio_padding(self, image, label):
        # pdb.set_trace()
        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        ratio = tf.math.round((w / h) * 3)
        ratio = tf.cast(ratio, dtype=tf.int32)

        ratio = tf.cond(tf.equal(ratio, 0),
                        lambda: 1,
                        lambda: ratio)

        ratio = tf.cond(tf.greater(ratio, self.max_ratio),
                        lambda: self.max_ratio,
                        lambda: ratio)

        # ratio = tf.cond(ratio == 0, lambda: tf.constant(1), lambda: tf.constant(ratio))

        # if ratio == 0.0:
        #     ratio = 1
        # if ratio > self.max_ratio:
        #     ratio = self.max_ratio

        h_new = 150
        w_new = h_new * ratio

        image_resize = tf.image.resize(image, (h_new, w_new))
        image_padding = tf.image.resize_with_pad(image_resize, target_height=h_new, target_width=h_new * self.max_ratio)

        return image_padding, label

    def parse_tfrecord(self, example):
        res = tf.io.parse_single_example(example, self.keys_to_features)
        image = tf.cast(tf.io.decode_jpeg(res['image/encoded'], 3), tf.float32) / 255.0
        label = tf.cast(res['image/class'], tf.float32)
        return image, label

    def load_tfrecord(self, repeat, batch_size, buffer_size=500):
        dataset = tf.data.TFRecordDataset(self.record_path)
        dataset = dataset.map(self.parse_tfrecord)
        dataset = dataset.map(self._process_image_ratio_padding)
        if repeat:
            dataset = dataset.repeat()
        self.dataset = dataset.batch(batch_size)
        self.batch_size = batch_size
        self.iterator = iter(self.dataset)

    def next_batch(self):
        return self.iterator.get_next()

    def __len__(self):
        from matplotlib import pyplot as plt
        assert self.dataset is not None, "Please load record after get leng of data"
        count = 0
        for x, y in tqdm(self.dataset):
            print(x.shape)
            # print(x[1, :])
            image = x[1, :].numpy()
            print(image.shape)
            plt.imshow(image)
            plt.show()

            count += 1
        return count * self.batch_size
