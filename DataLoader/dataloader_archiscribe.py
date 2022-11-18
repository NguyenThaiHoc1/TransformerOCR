import pdb

from tqdm import tqdm
import random
import tensorflow as tf
from DataLoader.dataloader import Dataloader
from DataLoader.TFSample.tfsample_archiscribe_corpus import TFSampleArchiscribeCorpus
from utils.function_helpers import read_pickle_file
import numpy as np
import torch


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Dataset(object):
    def __init__(self, record_path):
        self.record_path = record_path
        zero = tf.zeros([1], dtype=tf.int64)
        self.keys_to_features = {
            'image/img_encoded': tf.io.FixedLenFeature([], tf.string),
            'image/width': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            'image/height': tf.io.FixedLenFeature([1], tf.int64, default_value=zero),
            'image/token_ids': tf.io.FixedLenFeature([32], tf.int64),
            'image/text': tf.io.FixedLenFeature([], tf.string),
        }
        self.dataset = None
        self.iterator = None
        self.batch_size = None

        self.max_ratio = None

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create a mask to hide padding and future words, which are represented as 0 in the mask.
        """
        from torch.autograd import Variable
        tgt_mask = (tgt != pad)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        tgt_mask = tgt_mask.squeeze(0)  # the subsequent returned value has the shape of (1, N, N)
        return tgt_mask.numpy()

    @tf.function
    def _process_image_ratio_padding_black_right_image(self, image, label):

        def _py_function_process(bg_image, resize_image):
            bg_image_numpy = bg_image.numpy()
            resize_image_numpy = resize_image.numpy()
            bg_image_numpy[0:0 + resize_image_numpy.shape[0], 0:0 + resize_image_numpy.shape[1]] = resize_image_numpy
            return bg_image_numpy

        def _py_function_mask(label):
            grouth_truth = label.numpy()
            decode_in = grouth_truth[:-1]
            decode_out = grouth_truth[1:]
            tgt_mask = self.make_std_mask(torch.tensor(decode_in), 0)
            return tf.convert_to_tensor(decode_in, dtype=tf.uint8), \
                   tf.convert_to_tensor(decode_out, dtype=tf.uint8), \
                   tf.convert_to_tensor(tgt_mask, dtype=tf.bool)

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

        h_new = 150
        w_new = h_new * ratio
        image_resize = tf.image.resize(image, (h_new, w_new))
        background_padding = tf.zeros([h_new, h_new * self.max_ratio, 3], dtype=tf.float32)
        new_tensor = tf.py_function(_py_function_process, inp=[background_padding, image_resize], Tout=tf.float32)

        # create encoder mask for image
        encoder_mask = tf.concat([tf.ones([ratio]), tf.zeros([self.max_ratio - ratio])], axis=0)
        encoder_mask = tf.equal(encoder_mask, 1)[tf.newaxis, :]

        # label
        decode_in, decode_out, decode_mask = tf.py_function(_py_function_mask, inp=[label], Tout=[tf.uint8,
                                                                                                  tf.uint8,
                                                                                                  tf.bool])
        return tf.convert_to_tensor(new_tensor, dtype=tf.float32), label, encoder_mask, decode_mask

    @tf.function
    def _process_image_ratio_padding_no_black_right_image(self, image, label):
        def create_look_ahead_mask(size):
            mask = 1 - tf.linalg.band_part(tf.ones([size, size]), -1, 0)
            mask = tf.equal(mask, 0)
            return mask

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

        h_new = 150
        w_new = h_new * ratio
        image_resize = tf.image.resize(image, (h_new, w_new))

        image_padding = tf.image.resize_with_pad(image_resize, target_height=h_new, target_width=h_new * self.max_ratio)
        encoder_mask = None  # thuong se di theo ratio để tính nơi có từ và padding để phân tách

        # deocde label mask
        decode_in = None
        deocde_out = None
        size = tf.shape(label)[0]
        decode_mask = create_look_ahead_mask(size=size)
        return image_padding, label, encoder_mask, decode_mask

    def parse_tfrecord(self, example):
        res = tf.io.parse_single_example(example, self.keys_to_features)
        image = tf.io.decode_raw(res['image/img_encoded'], tf.uint8)
        image = tf.reshape(image, [150, 600, 3])
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.cast(res['image/token_ids'], tf.int64)
        return image, label

    def load_tfrecord(self, repeat, batch_size, buffer_size=10240, max_ratio=8,
                      with_padding_type="padding_without_right"):

        self.max_ratio = max_ratio

        dataset = tf.data.TFRecordDataset(self.record_path)
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=43)
        dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if with_padding_type == "padding_have_right":
            dataset = dataset.map(self._process_image_ratio_padding_black_right_image)

        elif with_padding_type == "padding_without_right":
            dataset = dataset.map(self._process_image_ratio_padding_no_black_right_image)

        if repeat:
            dataset = dataset.repeat()
        self.dataset = dataset.batch(batch_size)
        self.batch_size = batch_size
        self.iterator = iter(self.dataset)

    def next_batch(self):
        assert self.iterator is not None, "Please loading tfrecord"
        return self.iterator.get_next()

    def __len__(self):
        assert self.dataset is not None, "Please load record after get leng of data"
        count = 0
        for idx, (image, label, encode_mask, decode_mask) in enumerate(self.dataset):
            count += 1
            break
        return count * self.batch_size


class DataloaderArchiscribeCorPus(Dataloader):
    samples = []

    @staticmethod
    def activate(path_output_tfrecord):
        with tf.io.TFRecordWriter(path=path_output_tfrecord) as writer:
            for dict_info in tqdm(DataloaderArchiscribeCorPus.samples):
                img_encoded = dict_info['img_encoded'].tobytes()
                width = dict_info['width']
                height = dict_info['height']
                token_ids = [int(token) for token in dict_info['token_ids']]
                text = dict_info['text'].encode('utf-8')
                sample = TFSampleArchiscribeCorpus.create(
                    img_encoded=img_encoded,
                    width=width,
                    height=height,
                    token_ids=token_ids,
                    text=text
                )
                writer.write(record=sample.SerializeToString())

    @staticmethod
    def create(path_dataset):
        for pickle_file in path_dataset.glob('*.pkl'):
            data = read_pickle_file(str(pickle_file))
            DataloaderArchiscribeCorPus.samples.append(data)

        random.shuffle(DataloaderArchiscribeCorPus.samples)
