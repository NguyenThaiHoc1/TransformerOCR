import argparse
import glob
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from DataLoader.dataloader_archiscribe import Dataset as dataset_archiscribe
from DataLoader.dataloader_fsns import Dataset as dataset_fsns


def parser_generator():
    """
    E:\FSNS\FNFS-Dataset\FNFS\test\

    :return:
    """
    parser = argparse.ArgumentParser(description="Compress extract features raw.")
    parser.add_argument('--path_tfrec',
                        default=r"D:\hoc-nt\OCR_core\DatasetTFrecord\archiscribe-corpus\all_archiscribe_full.tfrec",
                        type=str,
                        required=False,
                        help="Path dir which contain dictionary.")
    parser.add_argument('--type_data', default="archiscribe",
                        type=str,
                        required=False,
                        help="Type dataset")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_generator()
    path_tfrec = Path(args.path_tfrec)
    name_dataset = args.type_data

    if name_dataset == 'archiscribe':
        cl_dataset = dataset_archiscribe(record_path=str(path_tfrec))
        cl_dataset.load_tfrecord(batch_size=1, repeat=False, max_ratio=15, with_padding_type="padding_without_right")
        num_image = len(cl_dataset)

        print(f"{args.type_data} :{num_image}")
    elif name_dataset == 'fsns':
        dataset_path_train = str(path_tfrec)
        list_files_tfrec_train = dataset_path_train  # glob.glob(os.path.join(dataset_path_train, '*'))
        cl_dataset = dataset_fsns(record_path=list_files_tfrec_train)
        cl_dataset.load_tfrecord(batch_size=16, repeat=False)
        num_image = len(cl_dataset)

        print(f"{args.type_data}: {num_image}")

    # metrics_names = ['acc', 'pr']
    #
    # progbar = tf.keras.utils.Progbar(312, stateful_metrics=metrics_names)
    #
    # count = 0
    # for idx, data in enumerate(cl_dataset.dataset):
    #     values = [('acc', np.random.random(1)), ('pr', np.random.random(1))]
    #
    #     progbar.add(4, values=values)
    #     count += 1
    #
    # print(f"The amount of dataset 1: {count}")
    #
    # count = 0
    # for input, label in tqdm(cl_dataset.dataset):
    #     count += 1
    #
    # print(f"The amount of dataset 2: {count}")

    # for idx in range(step):
    #     image, label = cl_dataset.next_batch()
    #     print(label)
    #     print(label.shape)
    #     break
