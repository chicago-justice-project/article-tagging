import pytest
import numpy as np

import tagnews

def test_binary_stemmed_model():
    tagger = tagnews.crimetype.tag.Tagger()
    computed = tagger.tagtext_proba(('This is an article about drugs and'
                                     ' gangs. Copyright Kevin Rose.'))

    expected_values = np.array(
        [0.97203727,  0.88789787,  0.2082669 ,  0.14037048,  0.10828103,
         0.09304668,  0.08045448,  0.07426908,  0.06969677,  0.06545745,
         0.06261077,  0.05765306,  0.05581387,  0.05180351,  0.04338474,
         0.03787152,  0.0372484 ,  0.03544577,  0.03451235,  0.03348183,
         0.03332619,  0.03229261,  0.02849745,  0.02814688,  0.02736295,
         0.02692241,  0.02479671,  0.02193516,  0.02170981,  0.02148271,
         0.02139072,  0.02041881,  0.01952591,  0.01899151,  0.01520583,
         0.0151583 ,  0.00705791,  0.00525071])

    np.testing.assert_array_almost_equal(computed.values,
                                         expected_values,
                                         decimal=4)

    expected_columns = [
        'GANG', 'DRUG', 'VIOL', 'IMMG', 'UNSPC', 'REEN', 'PARL', 'CPS',
        'GUNV', 'ILSC', 'IPRA', 'DUI', 'CCJ', 'GLBTQ', 'BEAT', 'POLM',
        'ILSP', 'CCSP', 'CPLY', 'ENVI', 'OEMC', 'TASR', 'CPBD', 'CPD',
        'HOMI', 'JUVE', 'ARSN', 'BURG', 'SEXA', 'IDOC', 'SAO', 'CPUB',
        'POLB', 'PROB', 'ROBB', 'FRUD', 'DOMV', 'CCCC'
    ]

    assert computed.index.values.tolist() == expected_columns
