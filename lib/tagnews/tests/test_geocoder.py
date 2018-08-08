import numpy as np
import pandas as pd

import tagnews


class Test_GeoCoder():
    @classmethod
    def setup_class(cls):
        cls.model = tagnews.GeoCoder()

    def test_extract_geostrings(self):
        self.model.extract_geostrings(
            ('This is example article text with a location of'
             ' 55th and Woodlawn where something happened.')
        )

    def test_extract_geostring_probs(self):
        article = ('This is example article text with a location of'
                   ' 55th and Woodlawn where something happened.')
        words, probs = self.model.extract_geostring_probs(article)
        max_prob = probs.max()
        max_word = words[np.argmax(probs)]
        geostrings = self.model.extract_geostrings(article,
                                                   prob_thresh=max_prob-0.001)
        assert max_word in [word for geostring in geostrings for word in geostring]

    def test_extract_geostring_probs_word_not_in_glove(self):
        """
        Regression test for issue #105.
        """
        article = '___1234567890nonexistent0987654321___'
        words, probs = self.model.extract_geostring_probs(article)

    def test_lat_longs_from_geostring_lists(self):
        geostring_lists = [['5500', 'S', 'Woodlawn'], ['100', 'N.', 'Wacker'], ['thigh']]
        coords, scores = self.model.lat_longs_from_geostring_lists(
            geostring_lists, sleep_secs=0.0
        )

        assert coords.shape[0] == len(geostring_lists) == len(scores)

    def test_community_areas(self):
        # Approximately 55th and Woodlawn, which is in Hyde Park.
        coords = pd.DataFrame([[41.793465, -87.596930]],
                              columns=['lat', 'long'])
        com_area = self.model.community_area_from_coords(coords)
        assert com_area == ['HYDE PARK']
