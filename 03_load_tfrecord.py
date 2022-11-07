import argparse
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from DataLoader.dataloader_archiscribe import Dataset as dataset_archiscribe


def parser_generator():
    parser = argparse.ArgumentParser(description="Compress extract features raw.")
    parser.add_argument('--path_tfrec', type=str, required=True, help="Path dir which contain dictionary.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_generator()
    path_tfrec = Path(args.path_tfrec)

    cl_dataset = dataset_archiscribe(record_path=str(path_tfrec))
    cl_dataset.load_tfrecord(batch_size=16)

    step = 2
    for idx in range(step):
        image, label = cl_dataset.next_batch()
        print(label)
        print(label.shape)
        break
