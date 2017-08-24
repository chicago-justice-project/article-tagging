import os

import pytest
import numpy as np

from tagnews.utils import load_data

_data_exists = os.path.isfile(os.path.join(load_data.__data_folder,
                                           'newsarticles_article.csv'))

@pytest.mark.filterwarnings('ignore')
@pytest.mark.skipif(not _data_exists,
                    reason='Data must be downloaded to load!')
def test_load_data():
    df = load_data.load_data()

    # Just assert things about article with ID 12345
    row = df.loc[12345, :]
    assert row['CCCC'] == 1
    assert row['CCJ'] == 0
    assert (row['bodytext'].split('\n')[0]
            == "![Larry E. Price (Sheriff's photo)][1]")
