from . import utils
from . import crimetype

from .crimetype.tag import Tagger
from .utils.load_data import load_data
from .utils.load_data import load_ner_data
from .utils.load_vectorizer import load_glove

__version__ = '0.6.0'

def test(verbosity=None, **kwargs):
    """run the test suite"""

    import pytest

    args = kwargs.pop('argv', [])

    if verbosity:
        args += ['-' + 'v' * verbosity]

    return pytest.main(args, **kwargs)


test.__test__ = False  # pytest: this function is not a test
