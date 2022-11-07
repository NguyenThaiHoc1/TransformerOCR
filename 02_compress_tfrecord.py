import argparse
from pathlib import Path
from DataLoader.dataloader_archiscribe import DataloaderArchiscribeCorPus


def parser_generator():
    parser = argparse.ArgumentParser(description="Compress extract features raw.")
    parser.add_argument('--path_dir', type=str, required=True, help="Path dir which contain dictionary.")
    parser.add_argument('--path_out', type=str, required=True, help="Path dir which contain dictionary.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_generator()
    path_dir = Path(args.path_dir)
    path_out = Path(args.path_out)

    DataloaderArchiscribeCorPus.create(path_dataset=path_dir)
    DataloaderArchiscribeCorPus.activate(path_output_tfrecord=str(path_out))
