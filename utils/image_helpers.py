import logging
import numpy as np
from PIL import Image


def read_image_from_file(path_file):
    logging.info("\tReading image ...")
    image = Image.open(path_file)  # open(path_file, 'rb').read()
    image = image.convert('RGB')
    image = image.resize((600, 150))
    logging.info("\tRead image done.")
    return np.asarray(image)


def read_text_from_file(path_file):
    with open(path_file, 'r', encoding='utf-8') as f:
        data = f.read().splitlines()[0]
    return data
