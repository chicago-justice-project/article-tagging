import load_data as ld
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import tensorflow as tf
import collections
import nltk

import pickle



LOAD_INDEXED = False

df = ld.load_data()

# adapted from tensorflow github:
# .../tensorflow/examples/tutorials/word2vec/word2vec_basic.py
def build_dataset(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words()).most_common(vocabulary_size - 1))
    global len_all_words
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = np.zeros(len_all_words)
    for i, word in enumerate(words()):
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data[i] = index
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def all_words():
    global len_all_words
    len_all_words = 0
    for txt in df['bodytext']:
        tokens = nltk.tokenize.word_tokenize(txt.lower())
        for t in tokens:
            yield t
            len_all_words += 1


def pickle_indexed(data, count, dictionary, reverse_dictionary):
    with open('../data/tf_dataset.pkl', 'wb') as f:
        data_set = {'data': data,
                    'count': count,
                    'dictionary': dictionary,
                    'reverse_dictionary': reverse_dictionary}
        pickle.dump(data_set, f) #~400 MB


def unpickle_indexed():
    with open('../data/tf_dataset.pkl', 'rb') as f:
        return pickle.load(f)

if LOAD_INDEXED:
    data_set = unpickle_indexed()
    data = data_set['data']
    count = data_set['count']
    dictionary = data_set['dictionary']
    reverse_dictionary = data_set['reverse_dictionary']
    del data_set
else:
    vocabulary_size = 50000
    data, count, dictionary, reverse_dictionary = build_dataset(all_words, vocabulary_size)
    pickle_indexed(data, count, dictionary, reverse_dictionary)



data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buff = collections.deque(maxlen=span)
    for _ in range(span):
        buff.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buff
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buff[skip_window]
            labels[i * num_skips + j, 0] = buff[target]
        buff.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels
