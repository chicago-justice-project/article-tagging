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

df = ld.load_data()

crime_df = df.ix[df.loc[:, 'OEMC':'TASR'].any(1), :]
print(crime_df.shape)
crime_df = crime_df.append(df.ix[~df['relevant'], :].sample(n=3000, axis=0))
print(crime_df.shape)

idx = np.random.permutation(crime_df.shape[0])
trn = crime_df.iloc[idx[:int(crime_df.shape[0] * 0.7)], :]
tst = crime_df.iloc[idx[int(crime_df.shape[0] * 0.7):], :]
print(trn.shape)
print(tst.shape)

# vectorize data
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


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

print(pd.DataFrame(
    clf.predict_proba(vectorizer.transform(['marijuana'])),
    columns=df.columns[7:]
).T.sort_values(0, ascending=False))


import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
