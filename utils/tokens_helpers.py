import re
import logging
import tensorflow as tf
from transformers import AutoTokenizer


def get_vocab_from_huggingface(name_model):
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    return len(tokenizer.vocab)


def get_vocab_from_file(filename, null_character=u'\u2591'):
    """Reads a charset definition from a tab separated text file.
      charset file has to have format compatible with the FSNS dataset.
      Args:
        filename: a path to the charset file.
        null_character: a unicode character used to replace '<null>' character. the
          default value is a light shade block '░'.
      Returns:
        a dictionary with keys equal to character codes and values - unicode
        characters.
      """
    pattern = re.compile(r'(\d+)\t(.+)')
    charset = {}
    with tf.io.gfile.GFile(filename) as f:
        for i, line in enumerate(f):
            m = pattern.match(line)
            if m is None:
                logging.warning('incorrect charset file. line #%d: %s', i, line)
                continue
            code = int(m.group(1))
            char = m.group(2)
            if char == '<nul>':
                char = null_character
            charset[code] = char
    return charset
