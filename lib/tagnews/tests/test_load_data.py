import os
import shutil

import pytest

import tagnews


class Test_LoadData():
    @staticmethod
    def setup_method():
        os.makedirs('./tmp/', exist_ok=True)

    @staticmethod
    def teardown_method():
        shutil.rmtree('./tmp/', ignore_errors=True)

    def test_load_data(self):
        df = tagnews.load_data()
        assert df.size

    def test_load_data_nrows(self):
        df = tagnews.load_data(nrows=2)
        assert df.size

    def test_subsample_and_resave(self):
        tagnews.utils.load_data.subsample_and_resave('./tmp/', n=1)

    def test_subsample_and_resave_raises_on_matching_folders(self):
        with pytest.raises(RuntimeError):
            tagnews.utils.load_data.subsample_and_resave(
                './tmp/', input_folder='./tmp/'
            )


class Test_LoadGlove():
    def test_load_glove(self):
        glove = tagnews.load_glove('tagnews/data/glove.6B.50d.txt')
        glove.loc['murder']
