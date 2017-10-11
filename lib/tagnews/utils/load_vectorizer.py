import numpy as np
import sklearn.preprocessing

def load_glove(vectors_file, normalize=True):
    """
    Load a GloVe formatted file, which is simply of the format
        <word_0><space><vec_0,0><space><vec_0,1><space>...<newline>
        <word_1><space><vec_1,0><space><vec_1,1><space>...<newline>
        ...
    See https://github.com/stanfordnlp/GloVe for more info.

    Inputs:
        vectors_file: path to file that contains GloVe formatted word
            vectors.
        normalize: Should the word vectors be normalized? See
            https://stats.stackexchange.com/questions/177905/ for
            a good discussion on the topic.

    Retuns:
        w: NxM word vectors matrix
        vocab: dictionary of the "word -> index into w" mapping
        inverse_vocab: dictionary of the "index into w -> word" mapping
    """

    vocab = {}
    inverse_vocab = {}
    with open(vectors_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            word = line.rstrip().split(' ')[0]
            vocab[word] = i
            inverse_vocab[i] = word

    vec_size = len(line.rstrip().split(' ')) - 1

    w = np.zeros((len(vocab), vec_size), dtype=np.float32)
    with open(vectors_file, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            w[i] = np.array([float(x) for x in line.rstrip().split(' ')[1:]])

    if normalize:
        w = sklearn.preprocessing.normalize(w)

    return w, vocab, inverse_vocab
