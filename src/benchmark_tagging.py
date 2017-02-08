from __future__ import division

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


def benchmark(clf_factory, X, Y, clf_params_dict=None, k=4):
    """
    benchmark a classifier on preprocessed data.

    Parameters
    ----------
    clf_factory :
        Function which returns a classifier. Needs to implement
        a `fit` method and a `predict` method. The parameters
        clf_params will be passed into this function.
    X : NxM matrix of features
    Y : NxL matrix of features
    clf_params_dict :
        dictionary of parameters passed to the classifier factory.
        If None, no parameters are passed.
    k : how many folds to use for cross validation
    """
    if clf_params_dict is None:
        clf_params_dict = {}

    fold_indexes = get_kfold_split(X.shape[0], k)
    acc = []

    for idx_trn, idx_tst in fold_indexes:
        clf = clf_factory(**clf_params_dict)

        x_trn = X[idx_trn, :]
        y_trn = Y[idx_trn, :]

        x_tst = X[idx_tst, :]
        y_tst = Y[idx_tst, :]

        clf.fit(x_trn, y_trn)
        y_hat = clf.predict(x_tst)

        acc.append((np.sum(y_tst == y_hat)) / float(y_tst.size))
        # TODO: acc is just a placeholder, replace with TPR, FPR, etc.

    return acc
