import os
import glob
import time
import pandas as pd
import numpy as np

from .. import utils

from contextlib import ExitStack, redirect_stderr
import os

with ExitStack() as stack:
    null_stream = open(os.devnull, "w")
    stack.enter_context(null_stream)
    stack.enter_context(redirect_stderr(null_stream))
    import keras

"""
Contains the CrimeTags class that allows tagging of articles.
"""

MODEL_LOCATION = os.path.join(os.path.split(__file__)[0],
                              os.path.join('models', 'lstm', 'saved'))


def load_model(location=MODEL_LOCATION):
    """
    Load a model from the given folder `location`.
    There should be at least one file named model-TIME.pkl and
    a file named vectorizer-TIME.pkl inside the folder.

    The files with the most recent timestamp are loaded.
    """
    models = glob.glob(os.path.join(location, 'weights*.hdf5'))
    if not models:
        raise RuntimeError(('No models to load. Run'
                            ' "python -m tagnews.geoloc.models.'
                            'lstm.save_model"'))

    model = keras.models.load_model(models[-1])

    return model


class GeoCoder():
    def __init__(self):
        self.model = load_model()
        self.glove = utils.load_vectorizer.load_glove(os.path.join(os.path.split(__file__)[0],
                                                                   '../data/glove.6B.50d.txt'))


    def pre_process(self, s):
        words = s.split() # split along white space.
        data = pd.concat([pd.DataFrame([[w[0].isupper()] if w else [False] for w in words]),
                          self.glove.loc[words].fillna(0).reset_index(drop=True)],
                         axis='columns')
        return words, np.expand_dims(data, axis=0)


    def extract_geostring_probs(self, s):
        words, data = self.pre_process(s)
        probs = self.model.predict(data)[0][:,1]
        return words, probs


    def extract_geostrings(self, s, prob_thresh=0.5):
        words, probs = self.extract_geostring_probs(s)
        above_thresh = probs >= prob_thresh

        words = ['filler'] + words + ['filler']
        above_thresh = np.concatenate([[False], above_thresh, [False]]).astype(np.int32)
        switch_ons = np.where(np.diff(above_thresh) == 1)[0] + 1
        switch_offs = np.where(np.diff(above_thresh) == -1)[0] + 1

        geostrings = []
        for on, off in zip(switch_ons, switch_offs):
            geostrings.append(words[on:off])

        return geostrings
