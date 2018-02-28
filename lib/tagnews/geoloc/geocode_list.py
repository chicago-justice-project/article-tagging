import geocoder
import time


# don't make more than 1 request per second
last_request_time = 0

def get_lat_longs_from_geostrings(geostring_list, provider='osm'):
    """
    Geo-code each geostring in `geostring_list` into lat/long values.
    Also return the full response from the geocoding service.

    Inputs
    ------
    geostring_list : list of strings
        The list of geostrings to geocode into lat/longs.

    Returns
    -------
    lat_longs : list of tuples
        The length `n` list of lat/long tuple pairs.
    full_responses : list
        The length `n` list of the full responses from the geocoding service.
    """
    global last_request_time

    providers = ['arcgis', 'google', 'yandex', 'geocodefarm', 'osm']
    assert provider in providers, \
        'I\'m sorry Dave, I\'m afraid I can\'t do that. \
        Please choose a provider from {}!'.format(' or '.join(providers))

    full_responses = []
    for addr_str in geostring_list:
        time_since_last_request = time.time() - last_request_time
        if time_since_last_request < 1:
            time.sleep((1 - time_since_last_request) + 0.1)
        g = getattr(geocoder, provider)(addr_str)
        full_responses.append(g)

    lat_longs = [g.latlng for g in full_responses]
    return lat_longs, full_responses
