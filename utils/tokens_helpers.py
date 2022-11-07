from transformers import AutoTokenizer


def get_vocab_from_huggingface(name_model):
    tokenizer = AutoTokenizer.from_pretrained(name_model)
    return len(tokenizer.vocab)
