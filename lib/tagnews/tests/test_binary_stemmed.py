import numpy as np
import pandas as pd

import tagnews

def test_binary_stemmed_model():
    tagger = tagnews.crimetype.tag.Tagger()
    computed = tagger.tagtext_proba(('This is an article about drugs and'
                                     ' gangs. Copyright Kevin Rose.'))

    expected_values = np.array(
        [0.960377, 0.865116, 0.138769, 0.125116, 0.074569, 0.064486, 0.062537,
         0.059940, 0.059029, 0.056022, 0.054410, 0.051550, 0.051503, 0.051264,
         0.044280, 0.042587, 0.039910, 0.036244, 0.031521, 0.031099, 0.030935,
         0.029413, 0.029372, 0.028972, 0.025446, 0.024925, 0.024656, 0.022577,
         0.022063, 0.021167, 0.020440, 0.018791, 0.018558, 0.018489, 0.017413,
         0.014794, 0.007131, 0.004737, ]
    )

    expected_columns = [
        'GANG', 'DRUG', 'IMMG', 'VIOL', 'REEN', 'GLBTQ', 'GUNV', 'UNSPC',
        'PARL', 'ILSC', 'BEAT', 'CCJ', 'IPRA', 'POLM', 'CPS', 'CPD', 'DUI',
        'ENVI', 'OEMC', 'CPBD', 'ILSP', 'TASR', 'CPLY', 'ARSN', 'JUVE',
        'FRUD', 'ROBB', 'BURG', 'CCSP', 'IDOC', 'HOMI', 'SEXA', 'POLB',
        'PROB', 'SAO', 'CPUB', 'DOMV', 'CCCC'
    ]

    expected = pd.Series(data=expected_values, index=expected_columns)

    np.testing.assert_array_almost_equal(computed[expected.index].values,
                                         expected.values,
                                         decimal=3)
