import argparse
import logging
from pathlib import Path
from utils.logging_helpers import setting_logging
from DataReader.archiscribe_reader import ArchiscribeDataReader


def parser_generator():
    parser = argparse.ArgumentParser(description="Compress extract features raw.")
    parser.add_argument('--path_dir', type=str, required=True, help="Path dir which contain dictionary.")
    parser.add_argument('--path_out_dir', type=str, required=True, help="Path dir which contain dictionary.")
    parser.add_argument('--type_data', type=str, required=True, help="Type of data.")
    parser.add_argument('--path_log', type=str, required=True, help="Log of process.")
    parser.add_argument('--max_length_sequence', type=int, required=True, help="Using in tokenizerof hugging face.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_generator()
    path_dir_dataset = Path(args.path_dir)
    path_out_pickle = Path(args.path_out_dir)

    path_log = Path(args.path_log)
    path_log.mkdir(exist_ok=True)
    setting_logging(path_logging=path_log)

    logging.info("*01_compress_extract_features_raw is processing ...")
    logging.info(f" Data type: {args.type_data}")
    datareader = None
    if args.type_data == 'archiscribe-corpus':
        datareader = ArchiscribeDataReader(path_dir=path_dir_dataset, path_out_dir=path_out_pickle,
                                           max_length_sequence=args.max_length_sequence)
    else:
        datareader = None

    assert datareader is not None, "Please checking data reader."
    datareader.read()
    logging.info(f"Done.")
