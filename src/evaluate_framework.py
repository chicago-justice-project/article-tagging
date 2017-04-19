import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.feature_extraction.text
import sklearn.multiclass
import sklearn.linear_model

import load_data as ld
import benchmark_tagging as bt

# plt.rcParams['figure.figsize'] = 12, 8

def create_example():
    # vectorize data
    from nltk import word_tokenize
    from nltk.stem import WordNetLemmatizer
    class LemmaTokenizer(object):
        def __init__(self):
            self.wnl = WordNetLemmatizer()
        def __call__(self, doc):
            return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=LemmaTokenizer(),
                                                                 binary=True)
    classifier_factory = lambda: sklearn.multiclass.OneVsRestClassifier(
        sklearn.linear_model.LogisticRegression()
    )

    df = ld.load_data()

    # TODO: Augment training data with not relevant

    crime_df = df.ix[df['relevant'], :]
    crime_df = crime_df.ix[crime_df.loc[:, 'OEMC':'TASR'].any(1), :]

    return (vectorizer, classifier_factory)


def evaluate(crime_df, vectorizer, classifier_factory,
             show_plot=True,
             predict_rand=True):
    """
    Evaluate the performance of a vectorizer/classifier combo.

    Args:
        crime_df : A pandas dataframe
    """
    bench_results = bt.benchmark(
        vectorizer.transform(crime_df['bodytext'].values),
        crime_df.loc[:, 'OEMC':'TASR'].values
    )

    column_labels = crime_df.loc[:, 'OEMC':'TASR'].columns.values.tolist()

    fpr = pd.DataFrame(bench_results['fpr'], columns=column_labels).T

    tpr = pd.DataFrame(bench_results['tpr'], columns=column_labels).T

    ppv = pd.DataFrame(bench_results['ppv'], columns=column_labels).T

    clf = bench_results['clfs'][0]

    if show_plot:
        f, axs = plt.subplots(3,1)
        tpr.mean(axis=1).plot(kind='bar', ax=axs[0])
        axs[0].set_ylabel('TPR')
        axs[0].set_xticklabels([])
        axs[0].set_ylim([0, 1])
        ppv.mean(axis=1).plot(kind='bar', ax=axs[1])
        axs[1].set_ylabel('PPV')
        axs[1].set_xticklabels([])
        axs[1].set_ylim([0, 1])
        (1 - fpr).mean(axis=1).plot(kind='bar', ax=axs[2])
        axs[2].set_ylabel('1 - FPR')
        axs[2].set_ylim([0, 1])
        plt.tight_layout()
        plt.show()

    if predict_rand:
        bt.predict_articles(clf, vectorizer, df)

    return {'fpr': fpr, 'tpr': tpr, 'ppv': ppv, 'clfs': bench_results['clfs']}


if __name__ == '__main__':
    evaluate(*create_example())
