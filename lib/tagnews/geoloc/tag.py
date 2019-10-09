from __future__ import division

import glob
import json
import os
import re
import time
from collections import namedtuple
from contextlib import ExitStack, redirect_stderr

import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape, Point

from tagnews.utils.neighborhoods import neighborhoods
from .. import utils

with ExitStack() as stack:
    null_stream = open(os.devnull, "w")
    stack.enter_context(null_stream)
    stack.enter_context(redirect_stderr(null_stream))
    import keras

"""
Contains the CrimeTags class that allows tagging of articles.
"""

MODEL_LOCATION = os.path.join(
    os.path.split(__file__)[0], os.path.join("models", "lstm", "saved")
)

COMMUNITY_AREAS_FILE = os.path.join(
    os.path.split(__file__)[0],
    "..",
    "data",
    "Boundaries - Community Areas (current).geojson",
)


def post_process(geostring):
    """
    Post process the geostring in a way that makes it more amenable to
    geocoding by the current geocoding provider GISgraphy.

    Inputs
    ------
    geostring : str
        The geostring to post process

    Returns
    -------
    processed_geostring : str
    """
    # Merge multiple whitespaces into one
    geostring = " ".join(geostring.split())

    # gisgraphy struggles with things like "55th and Woodlawn".
    # replace "...<number><number ender, e.g. th or rd> and..."
    # with two zeros.
    # \100 does not work correclty so we need to add a separator.
    geostring = re.sub(
        r"([0-9]+)(th|rd|st) and", r"\1<__internal_separator__>00 and", geostring
    )
    geostring = geostring.replace("<__internal_separator__>", "")

    # remove stopwords, only if they are internal, i.e.
    # the geostring doesn't start with "block ...".
    for stopword in ["block", "of", "and"]:
        geostring = geostring.replace(" {} ".format(stopword), " ")

    return geostring


_base_geocoder_url = (
    "http://ec2-34-228-58-223.compute-1.amazonaws.com" ":4000/v1/search?text={}"
)

GeocodeResults = namedtuple(
    "GeocodeResults",
    [
        "coords_raw",
        "full_responses_raw",
        "scores_raw",
        "coords_post",
        "full_responses_post",
        "scores_post",
    ],
)


def get_lat_longs_from_geostrings(
    geostring_list,
    post_process_f=None,
    sleep_secs=0,
    geocoder_url_formatter=_base_geocoder_url,
):
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
    sleep_secs : float
        How long to sleep between successive requests, in seconds.
    geocoder_url_formatter : str
        A string with a "{}" in it where the text should be input, e.g.
        "http://our-pelias.biz:4000/v1/search?text={}".

    Returns
    -------
    GeocodeResults : namedtuple
        A named tuple with the following fields:
        coords_raw : pandas.DataFrame
            The length `n` DataFrame of lat/long values. Values are NaN
            if the geocoder returned no results.
        full_responses_raw : list
            The length `n` list of the full responses from the geocoding
            service.
        scores_raw : numpy.array
            Numpy array of the confidence scores of the responses.
        coords_post : pandas.DataFrame
            The length `n` DataFrame of lat/long values. Values are NaN
            if the geocoder returned no results.
        full_responses_post : list
            The length `n` list of the full responses of the post-processed
            geostrings.
        scores_post : numpy.array
            Numpy array of the confidence scores of the responses.
    """
    if post_process_f is None:
        post_process_f = post_process

    def _geocode(lst):
        full_responses = []
        for addr_str in lst:
            try:
                g = json.loads(
                    requests.get(geocoder_url_formatter.format(addr_str)).text
                )
            except Exception:
                g = {}
            full_responses.append(g)
            time.sleep(sleep_secs)

        def _get_latlong(g):
            try:
                return g["features"][0]["geometry"]["coordinates"]
            except (KeyError, IndexError):
                return [np.nan, np.nan]

        def _get_confidence(g):
            try:
                return g["features"][0]["properties"]["confidence"]
            except (KeyError, IndexError):
                return np.nan

        coords = pd.DataFrame(
            [_get_latlong(g) for g in full_responses], columns=["long", "lat"]
        )
        coords = coords[["lat", "long"]]  # it makes me feel better, OK?
        scores = np.array([_get_confidence(g) for g in full_responses])

        return full_responses, coords, scores

    full_responses_raw, coords_raw, scores_raw = _geocode(geostring_list)

    full_responses_post, coords_post, scores_post = _geocode(
        [post_process_f(geo_s) for geo_s in geostring_list]
    )

    return GeocodeResults(
        coords_raw=coords_raw,
        full_responses_raw=full_responses_raw,
        scores_raw=scores_raw,
        coords_post=coords_post,
        full_responses_post=full_responses_post,
        scores_post=scores_post,
    )


def load_model(location=MODEL_LOCATION):
    """
    Load a model from the given folder `location`.
    There should be at least one file named model-TIME.pkl and
    a file named vectorizer-TIME.pkl inside the folder.

    The files with the most recent timestamp are loaded.
    """
    models = glob.glob(os.path.join(location, "weights*.hdf5"))
    if not models:
        raise RuntimeError(
            (
                "No models to load. Run"
                ' "python -m tagnews.geoloc.models.'
                'lstm.save_model"'
            )
        )

    model = keras.models.load_model(models[-1])

    return model


class GeoCoder:
    def __init__(self):
        self.model = load_model()
        self.glove = utils.load_vectorizer.load_glove(
            os.path.join(os.path.split(__file__)[0], "../data/glove.6B.50d.txt")
        )
        with open(COMMUNITY_AREAS_FILE) as f:
            d = json.load(f)
            self.com_areas = {
                f["properties"]["community"]: shape(f["geometry"])
                for f in d["features"]
            }

    def pre_process(self, s):
        """
        Takes in a string which is the text of an article and returns the tuple
        `(words, data)` where `words` is the list of words found and `data`
        is the 3D numpy array that contains the numeric data that can be used
        by the trained model.

        Inputs
        ------
        s : str
            Article text.

        Returns
        -------
        words : list of strings
            The words found in the article.
        data : 3D numpy.array
            Has shape (1, N, M) where N is the number of words and M
            is the size of the word vectors, currently M is 51.
        """
        words = s.split()  # split along white space.
        data = pd.concat(
            [
                pd.DataFrame([[w[0].isupper()] if w else [False] for w in words]),
                (self.glove.reindex(words).fillna(0).reset_index(drop=True)),
            ],
            axis="columns",
        )
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
        probs = self.model.predict(data)[0][:, 1]
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
            The list of extracted geostrings from the article text.
            Each word is kept separated in the list.
            Example:
                [['1300', 'W.', 'Halsted'], ['Ohio']]
        """
        words, probs = self.extract_geostring_probs(s)
        above_thresh = probs >= prob_thresh

        words = ["filler"] + words + ["filler"]
        probs = np.append(0, np.append(probs, 0))

        above_thresh = np.concatenate([[False], above_thresh, [False]]).astype(np.int32)
        switch_ons = np.where(np.diff(above_thresh) == 1)[0] + 1
        switch_offs = np.where(np.diff(above_thresh) == -1)[0] + 1

        geostrings = []
        probstrings = []
        for on, off in zip(switch_ons, switch_offs):
            geostrings.append(words[on:off])
            probstrings.append(probs[on:off])

        return geostrings, probstrings

    @staticmethod
    def lat_longs_from_geostring_lists(geostring_lists, **kwargs):
        """
        Get the latitude/longitude pairs from a list of geostrings as
        returned by `extract_geostrings`. Note that `extract_geostrings`
        returns a list of lists of words.

        Inputs
        ------
        geostring_lists : List[List[str]]
            A length-N list of list of strings, as returned by
            `extract_geostrings`.
            Example: [['5500', 'S.', 'Woodlawn'], ['1700', 'S.', 'Halsted']]
        **kwargs : other parameters passed to `get_lat_longs_from_geostrings`

        Returns
        -------
        coords : pandas.DataFrame
            A pandas DataFrame with columns "lat" and "long". Values are
            NaN if the geocoder returned no results.
        scores : numpy.array
            1D, length-N numpy array of the scores, higher indicates more
            confidence. This is our best guess after masssaging the scores
            returned by the geocoder, and should not be taken as any sort
            of absolute rule.
        """
        out = get_lat_longs_from_geostrings(
            [" ".join(gl) for gl in geostring_lists], **kwargs
        )

        return out.coords_post, out.scores_post

    def community_area_from_coords(self, coords):
        """
        Get the community area name that the coordinate lies in.

        Parameters
        ----------
        coords : pandas.DataFrame
            A pandas dataframe with columns "lat" and "long".

        Returns
        -------
        com_areas : List
            A list of community areas, one corresponding to each
            row of coords. An empty string indicates that the coord
            did not belong to any of the community areas.
        """
        out = []
        for _, coord in coords.iterrows():
            p = Point(coord["long"], coord["lat"])
            for com_name, com_shape in self.com_areas.items():
                if com_shape.contains(p):
                    out.append(com_name)
                    break
            else:
                out.append("")
        return out

    def best_geostring(self, extracted_strs_and_probs: tuple):
        """

        Parameters
        ----------
        extracted_strs_and_probs : 2-tuple
            A 2-tuple of two lists containing a list of extracted geostrings at index zero
                                and a list of extracted geostring probabilities at index one

        Returns
        -------
        2-tuple of one geostring of the best geostring
        OR False
        """
        consider = [[], []]
        for geostring, probs in zip(
            extracted_strs_and_probs[0], extracted_strs_and_probs[1]
        ):
            is_neighborhood = False
            for neighborhood in neighborhoods:
                if neighborhood.lower() in " ".join(geostring).lower():
                    is_neighborhood = True
            if is_neighborhood or len(geostring) >= 3:
                consider[0].append((geostring))
                consider[1].append((probs))
        if consider[0]:
            avgs = [sum(i) / len(i) for i in consider[1]]
            max_index = avgs.index(max(avgs))
            return consider[0][max_index]
        else:
            return ''

