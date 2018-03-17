from __future__ import division, print_function

import numpy as np
import pandas as pd


def get_kfold_split(N, k=4):
    """
    Create groups used for k-fold cross validation.

    Parameters
    ----------
    N : number of samples to split
    k : number of groups used for cross validation

    Returns
    -------
    List of (index_train, index_test) pairs
    """
    np.random.seed(2017)
    idx = np.random.permutation(N)
    index_pairs = [(np.ones(N).astype(np.bool),
                    np.zeros(N).astype(np.bool))
                   for _ in range(k)]

    for i, fold_idx in enumerate(np.array_split(idx, k)):
        index_pairs[i][0][fold_idx] = 0
        index_pairs[i][1][fold_idx] = 1

    return index_pairs


def benchmark(clf_factory, X, Y, clf_params_dict=None, k=4, verbose=False):
    """
    benchmark a classifier on preprocessed data.

    Parameters
    ----------
    clf_factory :
        Function which returns a classifier. Classifiers implement
        a `fit` method and a `predict` method. The parameters
        clf_params will be passed to clf_factory.
    X : NxM matrix of features
    Y : NxL matrix of binary values. Y[i,j] indicates whether or
        not the j'th tag applies to the i'th article.
    clf_params_dict :
        dictionary of parameters passed to the classifier factory.
        If None, no parameters are passed.
    k : how many folds to use for cross validation
    verbose : Should status be printed?
    """
    if clf_params_dict is None:
        clf_params_dict = {}

    L = Y.shape[1]

    fold_indexes = get_kfold_split(X.shape[0], k)
    acc = np.zeros(k)
    tpr = np.zeros((k, L))
    fpr = np.zeros((k, L))
    ppv = np.zeros((k, L))

    clfs = []
    for i, (idx_trn, idx_tst) in enumerate(fold_indexes):
        if verbose:
            print('step {} of {}...'.format(i, k), end='')

        clf = clf_factory(**clf_params_dict)

        x_trn = X[idx_trn, :]
        y_trn = Y[idx_trn, :]

        x_tst = X[idx_tst, :]
        y_tst = Y[idx_tst, :]

        clf.fit(x_trn, y_trn)
        y_hat = clf.predict_proba(x_tst)
        y_hat = y_hat > 0.5

        y_hat.dtype = np.int8
        y_tst.dtype = np.int8

        acc[i] = (np.sum(y_tst == y_hat)) / float(y_tst.size)
        for j in range(L):
            tpr[i, j] = np.sum(y_tst[:, j] & y_hat[:, j]) / np.sum(y_tst[:, j])
            fpr[i, j] = (np.sum(np.logical_not(y_tst[:, j]) & y_hat[:, j])
                         / np.sum(np.logical_not(y_tst[:, j])))
            ppv[i, j] = np.sum(y_tst[:, j] & y_hat[:, j]) / np.sum(y_hat[:, j])

        clfs.append(clf)

        if verbose:
            print('done')

    return {'acc': acc, 'tpr': tpr, 'fpr': fpr, 'ppv': ppv, 'clfs': clfs}


def predict_articles(clf, vectorizer, df, n=100, seed=1029384756):
    np.random.seed(seed)

    pd.set_option('display.max_columns', 100)
    pd.set_option('display.float_format', lambda x: '%.6f' % x)

    random_subset = np.random.choice(np.arange(df.shape[0]),
                                     size=n,
                                     replace=False)

    preds = clf.predict_proba(vectorizer.transform(
        df.iloc[random_subset, 3].values
    ))
    preds = pd.DataFrame(preds)
    preds.columns = df.loc[:, 'OEMC':'TASR'].columns

    for i, rand_i in enumerate(random_subset):
        s = 'Article ID: ' + str(df.index[rand_i])
        s += '\n' + df.iloc[rand_i, 3]
        s += '\n Predicted Tags: '
        s += str(preds.iloc[i, :].index[preds.iloc[i, :] > 0.5].values)
        s += '\n' + str(preds.iloc[i, :])
        s += '\n'
        filename = 'test-tag-' + str(df.index[rand_i]) + '.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(s)
