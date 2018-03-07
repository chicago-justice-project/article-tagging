import os
from collections import namedtuple
import glob
import time

import geocoder
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


def post_process(geostring):
    # TODO
    geostring += ' Chicago, Illinois'
    geostring.replace('block of ', '')
    return geostring


GeocodeResults = namedtuple('GeocodeResults', ['lat_longs_raw',
                                               'full_responses_raw',
                                               'lat_longs_post',
                                               'full_responses_post'])


def get_lat_longs_from_geostrings(geostring_list, post_process_f=None):
    """
    Geo-code each geostring in `geostring_list` into lat/long values.
    Also return the full response from the geocoding service.

    Inputs
    ------
    geostring_list : list of strings
        The list of geostrings to geocode into lat/longs.
    post_process_f : function
        The results are returned for both the raw geostrings being
        passed to the geocoder, and the results of
        `post_process_f(geostring)` being passed to the geocoder.

    Returns
    -------
    GeocodeResults : namedtuple
        A named tuple with the following fields:
        lat_longs_raw : list of tuples
            The length `n` list of lat/long tuple pairs or None.
        full_responses_raw : list
            The length `n` list of the full responses from the geocoding service.
        lat_longs_post : list of tuples
            The length `n` list of lat/long tuple pairs or None of the post-processed geostrings.
        full_responses_post : list
            The length `n` list of the full responses of the post-processed geostrings.
    """
    if post_process_f is None:
        post_process_f = post_process

    def _geocode(lst):
        full_responses = []
        for addr_str in geostring_list:
            g = geocoder.gisgraphy(addr_str)
            full_responses.append(g)
            time.sleep(0.5) # not technically required but let's be kind

        lat_longs = [g.latlng for g in full_responses]

        return full_responses, lat_longs

    full_responses_raw, lat_longs_raw = _geocode(geostring_list)

    geostring_list = [post_process_f(geostring) for geostring in geostring_list]
    full_responses_post, lat_longs_post = _geocode(geostring_list)



    return GeocodeResults(lat_longs_raw=lat_longs_raw,
                          full_responses_raw=full_responses_raw,
                          lat_longs_post=lat_longs_post,
                          full_responses_post=full_responses_post)


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
        """
        Takes in a string which is the text of an article and returns the tuple
        `(words, data)` where `words` is the list of words found and `data`
        is the 3D numpy array that contains the numeric data that can be used by
        the trained model.

        Inputs
        ------
        s : str
            Article text.

        Returns
        -------
        words : list of strings
            The words found in the article.
        data : 3D numpy.array
            Has shape (1, N, M) where N is the number of words and M is the size of the
            word vectors, currently M is 51.
        """
        words = s.split() # split along white space.
        data = pd.concat([pd.DataFrame([[w[0].isupper()] if w else [False] for w in words]),
                          self.glove.loc[words].fillna(0).reset_index(drop=True)],
                         axis='columns')
        return words, np.expand_dims(data, axis=0)


    def extract_geostring_probs(self, s):
        """
        Extract the probability that each word in s is part of a geostring.

        Inputs
        ------
        s : str
            Article text.

        Returns
        -------
        words : list of strings
            The words found in the article.
        probs : 1D numpy.array
            Has shape (N,) where N is the number of words.
        """
        if not s.strip():
            return [[], np.zeros((0,), dtype=np.float32)]
        words, data = self.pre_process(s)
        probs = self.model.predict(data)[0][:,1]
        return words, probs


    def extract_geostrings(self, s, prob_thresh=0.5):
        """
        Extract the geostrings from the article text.

        Inputs
        ------
        s : str
            Article text.
        prob_thresh : float, 0 <= prob_thresh <= 1
            The threshold on probability above which words will be
            considered as part of a geostring.
            DEFAULT: 0.5

        Returns
        -------
        geostrings : list of lists of strings
            The list of extracted geostrings from the article text. Each word is kept
            separated in the list.
            Examle:
                [['1300', 'W.', 'Halsted'], ['Ohio']]
        """
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
