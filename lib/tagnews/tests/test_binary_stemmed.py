import numpy as np
import pandas as pd

import tagnews

def test_binary_stemmed_model():
    tagger = tagnews.crimetype.tag.Tagger()
    computed = tagger.tagtext_proba(('This is an article about drugs and'
                                     ' gangs. Copyright Kevin Rose.'))

    expected_values = np.array(
        [0.974812, 0.899625, 0.246876, 0.133746, 0.106545, 0.104717,
         0.087247, 0.085050, 0.083862, 0.070980, 0.070646, 0.068562,
         0.056926, 0.049988, 0.046514, 0.045374, 0.042793, 0.041884,
         0.038661, 0.037252, 0.036225, 0.034859, 0.033933, 0.033609,
         0.028948, 0.025361, 0.025335, 0.024939, 0.023819, 0.022974,
         0.022706, 0.021454, 0.019619, 0.019261, 0.016800, 0.016401,
         0.009044, 0.006117]
    )

    expected_columns = [
        'GANG', 'DRUG', 'VIOL', 'IMMG', 'GUNV', 'REEN', 'UNSPC', 'PARL',
        'CPS', 'ILSC', 'IPRA', 'GLBTQ', 'CCJ', 'BEAT', 'DUI', 'ENVI',
        'POLM', 'ILSP', 'TASR', 'OEMC', 'HOMI', 'CPLY', 'ARSN', 'CPBD',
        'CCSP', 'BURG', 'JUVE', 'IDOC', 'SAO', 'PROB', 'CPUB', 'POLB',
        'CPD', 'FRUD', 'ROBB', 'SEXA', 'DOMV', 'CCCC'
    ]

    expected = pd.Series(data=expected_values, index=expected_columns)

    np.testing.assert_array_almost_equal(computed[expected.index].values,
                                         expected.values,
                                         decimal=3)
