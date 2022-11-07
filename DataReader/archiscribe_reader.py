"""
https://stackoverflow.com/questions/68031093/python-threadpoolexecutor-not-running-parallelly
"""
import os
import time
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from DataReader.datareader import DataReader
from transformers import AutoTokenizer
from utils.image_helpers import read_image_from_file, read_text_from_file


class ArchiscribeDataReader(DataReader):

    def __init__(self, path_dir, path_out_dir, max_length_sequence):
        super(ArchiscribeDataReader, self).__init__(path_dir=path_dir, path_out_dir=path_out_dir)
        self._tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')
        self._max_length_sequence = max_length_sequence
        self.pool_executor = ThreadPoolExecutor(os.cpu_count())

    def _encode_text(self, text):
        tokens = self._tokenizer.encode_plus(text, None,
                                             max_length=self._max_length_sequence,
                                             padding='max_length',
                                             add_special_tokens=True,
                                             return_token_type_ids=True,
                                             truncation=True)
        return tokens['input_ids']  # input_ids, token_type_ids, attention_mask

    def _process(self, path_file, path_txt, path_pkl):
        logging.info("\tProcess is starting ...")
        np_image = read_image_from_file(path_file=path_file)
        data_txt = read_text_from_file(path_file=path_txt)
        height, width = np_image.shape[:2]
        token_ids = self._encode_text(data_txt)
        logging.info("\tProcess done.")
        dict_info_result = {
            'img_encoded': np_image,
            'width': width,
            'height': height,
            'token_ids': token_ids,
            'text': data_txt
        }
        pickle.dump(dict_info_result, open(f'{path_pkl}.pkl', 'wb'))

    def read(self):
        logging.info("\tStarting datareader 'archiscribe-corpus' ...")
        list_dir_archiscribe = self.path.glob('**/*')
        features = []
        for dir_archiscribe in list_dir_archiscribe:
            list_files_image = dir_archiscribe.glob('*.png')
            for idx, file_image in enumerate(list_files_image):
                file_image_path = str(file_image)
                file_text_path = file_image_path.replace('.png', '.txt')
                file_name_pickle = str(self.path_out / file_image.stem)
                feature = self.pool_executor.submit(self._process, file_image_path, file_text_path, file_name_pickle)
                features.append(feature)
        logging.info("\tDatareader 'archiscribe-corpus' done.")
        wait(features)
        # pool_executor.shutdown(wait=True) or meethod to stop worker
