from tqdm import tqdm
import random
import tensorflow as tf
from DataLoader.dataloader import Dataloader
from DataLoader.TFSample.tfsample_archiscribe_corpus import TFSampleArchiscribeCorpus
from utils.function_helpers import read_pickle_file


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

    def parse_tfrecord(self, example):
        res = tf.io.parse_single_example(example, self.keys_to_features)
        image = tf.io.decode_raw(res['image/img_encoded'], tf.uint8)
        image = tf.reshape(image, [150, 600, 3])
        label = tf.cast(res['image/token_ids'], tf.int64)
        return image, label

    def load_tfrecord(self, repeat, batch_size, buffer_size=10240):
        self.dataset = tf.data.TFRecordDataset(self.record_path)
        self.dataset = self.dataset.shuffle(buffer_size=buffer_size, seed=43)
        self.dataset = self.dataset.map(self.parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.batch(batch_size)

        self.batch_size = batch_size
        if repeat:
            self.dataset = self.dataset.repeat()
        self.iterator = iter(self.dataset)

    def next_batch(self):
        assert self.iterator is not None, "Please loading tfrecord"
        return self.iterator.get_next()

    def __len__(self):
        assert self.dataset is not None, "Please load record after get leng of data"
        count = 0
        for idx, _ in enumerate(self.dataset):
            count += 1
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
