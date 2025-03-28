from . import utils
from . import crimetype

from .crimetype.tag import CrimeTags
from .senteval.eval import SentimentGoogler
from .geoloc.tag import GeoCoder, get_lat_longs_from_geostrings
from .utils.load_data import load_data
from .utils.load_vectorizer import load_glove

__all__ = [utils, crimetype, CrimeTags, GeoCoder, SentimentGoogler,
           get_lat_longs_from_geostrings, load_data, load_glove]

__version__ = '1.4.0'
