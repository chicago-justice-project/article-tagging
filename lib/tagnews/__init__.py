from . import utils
from . import crimetype

from .crimetype.tag import Tagger
from .utils.load_data import load_data

__version__ = '0.5.0'

def test(verbosity=None, **kwargs):
    """run the test suite"""

    import pytest

    args = kwargs.pop('argv', [])

    if verbosity:
        args += ['-' + 'v' * verbosity]

    return pytest.main(args, **kwargs)


test.__test__ = False  # pytest: this function is not a test
