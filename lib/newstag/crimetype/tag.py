import os
import pickle
import pandas as pd

# not used explicitly, but this needs to be imported like this
# for unpickling to work.
from ..utils.model_helpers import LemmaTokenizer

"""
Contains the Tagger class that allows tagging of articles.

This file can also be run as a module, with
`python -m newstag.crimetype.tag`
"""

MODEL_LOCATION = os.path.join(os.path.split(__file__)[0],
                              os.path.join('models', 'binary_stemmed_logistic'))

TAGS = ['OEMC', 'CPD', 'SAO', 'CCCC', 'CCJ', 'CCSP',
        'CPUB', 'IDOC', 'DOMV', 'SEXA', 'POLB', 'POLM',
        'GUNV', 'GLBTQ', 'JUVE', 'REEN', 'VIOL', 'BEAT',
        'PROB', 'PARL', 'CPLY', 'DRUG', 'CPS', 'GANG', 'ILSP',
        'HOMI', 'IPRA', 'CPBD', 'IMMG', 'ENVI', 'UNSPC',
        'ILSC', 'ARSN', 'BURG', 'DUI', 'FRUD', 'ROBB', 'TASR']


def load_model(location=MODEL_LOCATION):
    with open(os.path.join(location, 'model.pkl'), 'rb') as f:
        clf = pickle.load(f)

    with open(os.path.join(location, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    return clf, vectorizer


class Tagger():
    """
    Taggers let you tag articles. Neat!
    """
    def __init__(self, model_directory=MODEL_LOCATION):
        self.clf, self.vectorizer = load_model(model_directory)

    def tagtext_proba(self, text):
        """
        Compute the probability each tag applies to the given text.

        inputs:
            text: A python string.
        returns:
            pred_proba: A pandas series indexed by the tag name.
        """
        x = self.vectorizer.transform([text])
        y_hat = self.clf.predict_proba(x)
        preds = pd.DataFrame(y_hat)
        preds.columns = TAGS
        preds = preds.T.iloc[:,0].sort_values(ascending=False)
        return preds


    def tagtext(self, text, prob_thresh=0.5):
        """
        Tag a string with labels.

        inputs:
            text: A python string.
            prob_thresh: The threshold on probability at which point
                the tag will be applied.
        returns:
            preds: A list of tags that have > prob_thresh probability
                according to the model.
        """
        preds = self.tagtext_proba(text)
        return preds[preds > prob_thresh].index.values.tolist()


    def relevant(self, text, prob_thresh=0.05):
        """
        Determines whether given text is relevant or not. Relevance
        is defined as whether any tag has more than prob_thresh
        chance of applying to the text according to the model.

        inputs:
            text: A python string.
            prob_thresh: The threshold on probability that
                determines relevance. If no tags have >=
                prob_thresh of applying to the text, then
                the text is not relevant.
        returns:
            relevant: Boolean. Is the text "relevant"?
        """
        return len(self.tagtext(text, prob_thresh)) > 0
