from . import utils
from . import crimetype

from .crimetype.tag import CrimeTags
from .geoloc.tag import GeoCoder, get_lat_longs_from_geostrings
from .utils.load_data import load_data
from .utils.load_data import load_ner_data
from .utils.load_vectorizer import load_glove

__version__ = '1.0.2'
