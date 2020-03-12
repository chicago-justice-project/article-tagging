import numpy as np
import pandas as pd
import sklearn.preprocessing


def load_glove(vectors_file, normalize=False):
    """
    Load a GloVe formatted file, which is simply of the format

        <word_0><space><vec_0,0><space><vec_0,1><space>...<newline>
        <word_1><space><vec_1,0><space><vec_1,1><space>...<newline>
        ...
    See https://github.com/stanfordnlp/GloVe for more information.
    That link also has information on how to download the pre-trained
    word vectorizer models. If the file you download is compressed,
    you will need to uncompress it before using this function.

    Note that the loading speed and memory usage is highly depdendent
    on what model you use. The downloadable model "glove.840B.300d.txt"
    will take a few minutes to load and use 2.8 GB of memory, whereas the
    model "glove.6B.50d.txt" will take a few seconds and use < 200 MB
    of memory.

    Sample usage:

        >>> vectors = load_glove('tagnews/geoloc/glove.6B.50d.txt')
        >>> text = 'This is a sentence and stuff.'
        >>> # you should use an actual tokenizer for this step.
        >>> vectorized_text = vectors.loc[[word.lower()
        ...                                for word in text.split()]]
        >>> print(vectorized_text.shape)
            (6, 300)
        >>> k = 5
        >>> import numpy as np
        >>> def euc(word):
        ...   return np.sum((vectors.values-vectors.loc[word].values)**2.0, 1)
        ...
        >>> vectors.index[np.argpartition(euc('murder'), range(k))[:k]]

    Inputs:
        vectors_file: path to file that contains GloVe formatted word
            vectors.
        normalize: Should the word vectors be normalized? See
            https://stats.stackexchange.com/questions/177905/ for
            a good discussion on the topic.

    Retuns:
        vectors: NxM pandas dataframe whose rows are indexed by the word.
    """

    with open(vectors_file, 'r', encoding='utf-8') as f:
        for vocab_size, line in enumerate(f):
            pass
    vocab_size += 1

    vec_size = len(line.split(' ')) - 1
    vectors = np.zeros((vocab_size, vec_size), dtype=np.float32)
    words = np.empty(shape=(vocab_size), dtype=np.dtype('object'))

    with open(vectors_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.split(' ')
            words[i] = line[0]
            vectors[i] = [float(x) for x in line[1:]]

    vectors = pd.DataFrame(vectors, index=words, copy=False)
    vectors = vectors.loc[~vectors.index.duplicated()]

    if normalize:
        sklearn.preprocessing.normalize(vectors, copy=False)

    return vectors
