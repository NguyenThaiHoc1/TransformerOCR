import pickle


def read_pickle_file(path_file_pickle):
    with open(path_file_pickle, 'rb') as f:
        data = pickle.load(f)
    return data
