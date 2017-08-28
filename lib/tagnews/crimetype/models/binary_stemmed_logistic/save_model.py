from ....utils import load_data as ld
from ....utils.model_helpers import LemmaTokenizer
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.feature_extraction.text
import sklearn.multiclass
import sklearn.linear_model
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

df = ld.load_data()

crime_df = df.ix[df.loc[:, 'OEMC':'TASR'].any(1), :]
crime_df = crime_df.append(df.ix[~df['relevant'], :].sample(n=3000, axis=0))

idx = np.random.permutation(crime_df.shape[0])
trn = crime_df.iloc[idx[:int(crime_df.shape[0] * 0.7)], :]
tst = crime_df.iloc[idx[int(crime_df.shape[0] * 0.7):], :]

vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=LemmaTokenizer(),
                                                             binary=True)
X = vectorizer.fit_transform(trn['bodytext'].values)

Y = trn.loc[:, 'OEMC':'TASR'].values

clf = sklearn.multiclass.OneVsRestClassifier(
    sklearn.linear_model.LogisticRegression()
)

X = vectorizer.transform(crime_df['bodytext'].values)
Y = crime_df.loc[:, 'OEMC':'TASR'].values

clf.fit(X, Y)

import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
