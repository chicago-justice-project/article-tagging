import numpy as np
import pandas as pd
import sklearn.preprocessing

def load_glove(vectors_file, normalize=True):
    """
    Load a GloVe formatted file, which is simply of the format
        <word_0><space><vec_0,0><space><vec_0,1><space>...<newline>
        <word_1><space><vec_1,0><space><vec_1,1><space>...<newline>
        ...
    See https://github.com/stanfordnlp/GloVe for more info.

    Sample usage:
    >>> vectors = load_glove('tagnews/data/glove.840B.300d.txt')
    >>> text = 'This is a sentence and stuff.'
    >>> vectorized_text = vectors.loc[[word.lower() for word in text.split()]]
    >>> print(vectorized_text.shape)
        (300, 6)
    >>> k = 5
    >>> import numpy as np
    >>> def euc(word):
    ...     return np.sum((vectors.values - vectors.loc[word].values) ** 2.0, 1)
    >>> vectors.index[np.argpartition(euc('murder'), range(k))[:k]]

import tagnews
import datetime
print(datetime.datetime.now())
vectors = tagnews.utils.load_vectorizer.load_glove('tagnews/data/glove.840B.300d.txt', normalize=False)
print(datetime.datetime.now())
text = 'This is a sentence and stuff.'
vectorized_text = vectors.loc[[word.lower() for word in text.split()]]
print(vectorized_text.shape)
k = 5
import numpy as np
def euc(word):
    return np.sum((vectors.values - vectors.loc[word].values) ** 2.0, 1)

vectors.index[np.argpartition(euc('murder'), range(k))[:k]]

Index(['murder', 'murders', 'murdered', 'homicide', 'crime'], dtype='object')

    Inputs:
        vectors_file: path to file that contains GloVe formatted word
            vectors.
        normalize: Should the word vectors be normalized? See
            https://stats.stackexchange.com/questions/177905/ for
            a good discussion on the topic.

    Retuns:
        vectors: NxM pandas dataframe whose rows are indexed by the word.
    """

    vocab = {}
    duplicate_word_idxs = []
    num_duplicates = 0

    with open(vectors_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            word = line[:line.index(' ')]
            if word in vocab:
                duplicate_word_idxs.append(i)
                num_duplicates += 1
                continue
            vocab[word] = i - num_duplicates

    # As written, this is guaranteed to be sorted. But sorting
    # this will be cheap and good future proofing.
    duplicate_word_idxs = sorted(duplicate_word_idxs)
    skip_counter = 0
    vec_size = len(line.rstrip().split(' ')) - 1
    w = np.zeros((len(vocab), vec_size), dtype=np.float32)

    with open(vectors_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if skip_counter < num_duplicates and i == duplicate_word_idxs[skip_counter]:
                skip_counter += 1
                continue
            w[i - skip_counter] = [float(x) for x in line.split(' ')[1:]]

    if normalize:
        w = sklearn.preprocessing.normalize(w)

    vectors = pd.DataFrame(
        w,
        index=[item[0] for item in sorted(vocab.items(), key=lambda item: item[1])],
        copy=False
    )

    return vectors
