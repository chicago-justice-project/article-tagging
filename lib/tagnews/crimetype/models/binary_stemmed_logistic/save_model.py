import os
import time
import sys

from ....utils import load_data as ld
from ....utils.model_helpers import LemmaTokenizer
import numpy as np
import pandas as pd
import sklearn
import sklearn.feature_extraction.text
import sklearn.multiclass
import sklearn.linear_model
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

np.random.seed(1029384756)

if len(sys.argv) == 2:
    df = ld.load_data(nrows=int(sys.argv[1]))
elif len(sys.argv) == 1:
    df = ld.load_data()
else:
    raise Exception('BAD ARGUMENTS')

crime_df = df.ix[df.loc[:, 'OEMC':'TASR'].any(1), :]
crime_df = crime_df.append(
    df.ix[~df['relevant'], :].sample(n=min(3000, (~df['relevant']).sum()),
                                     axis=0)
)

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=LemmaTokenizer(),
                                                             binary=True,
                                                             max_features=40000)

clf = sklearn.multiclass.OneVsRestClassifier(
    sklearn.linear_model.LogisticRegression(verbose=1)
)

X = vectorizer.fit_transform(crime_df['bodytext'].values)
Y = crime_df.loc[:, 'OEMC':'TASR'].values

clf.fit(X, Y)

from ...tag import CrimeTags

crimetags = CrimeTags(clf=clf, vectorizer=vectorizer)

print(crimetags.tagtext_proba(('This is an article about drugs and'
                               ' gangs.')))

import pickle

curr_time = time.strftime("%Y%m%d-%H%M%S")

with open(os.path.join(os.path.split(__file__)[0],
                       'model-' + curr_time + '.pkl'), 'wb') as f:
    pickle.dump(clf, f)
with open(os.path.join(os.path.split(__file__)[0],
                       'vectorizer-' + curr_time + '.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)
