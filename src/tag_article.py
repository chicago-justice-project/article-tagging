import sys
import os
import pickle
import argparse

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd


MODEL_LOCATION = '../python_models/binary-stemmed-logistic/'

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

with open(os.path.join(MODEL_LOCATION, 'model.pkl'), 'rb') as f:
    clf = pickle.load(f)

with open(os.path.join(MODEL_LOCATION, 'vectorizer.pkl'), 'rb') as f:
    vectorizer = pickle.load(f)


def tag_string(s):
    tags = ['OEMC', 'CPD', 'SAO', 'CCCC', 'CCJ', 'CCSP',
     'CPUB', 'IDOC', 'DOMV', 'SEXA', 'POLB', 'POLM',
     'GUNV', 'GLBTQ', 'JUVE', 'REEN', 'VIOL', 'BEAT',
     'PROB', 'PARL', 'CPLY', 'DRUG', 'CPS', 'GANG', 'ILSP',
     'HOMI', 'IPRA', 'CPBD', 'IMMG', 'ENVI', 'UNSPC',
     'ILSC', 'ARSN', 'BURG', 'DUI', 'FRUD', 'ROBB', 'TASR']

    preds = clf.predict_proba(vectorizer.transform([s.replace('\n', ' ')]))
    preds = pd.DataFrame(preds)
    preds.columns = tags
    return preds


if __name__ == '__main__':
    if len(sys.argv) == 1:
        s = sys.stdin.read()
        preds = tag_string(s)
        preds = preds.sort_values(by=0, axis=1, ascending=False)
        for tag, prob in zip(preds.columns, preds.values[0]):
            print('{: >5}, {:.9f}'.format(tag, prob))
    else:
        for filename in sys.argv[1:]:
            with open(filename) as f_in:
                preds = tag_string(f_in.read())
            preds = preds.sort_values(by=0, axis=1, ascending=False)
            with open(filename + '.tagged', 'w') as f_out:
                for tag, prob in zip(preds.columns, preds.values[0]):
                    f_out.write('{: >5}, {:.9f}\n'.format(tag, prob))
