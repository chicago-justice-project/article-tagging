import geocoder
import time

def lat_longs_from_geo_strings(lst):
    lats_lons = []
    for addr_str in lst:
        g = geocoder.google(addr_str)
        if g.latlng is None:
            time.sleep(.5)
            lats_lons.extend(lat_longs_from_geo_strings([addr_str]))
        else:
            lats_lons.append(g.latlng)
    return lats_lons


def multi_option_lat_longs(lst, provider='arcgis'):
    providers = ['arcgis', 'google', 'yandex', 'geocodefarm', 'osm']
    assert provider in providers, \
        'I\'m sorry Dave, I\'m afraid I can\'t do that. \
        Please choose a provider from {}!'.format(' or '.join(providers))
    lats_lons = []
    for addr_str in lst:
        time.sleep(1)
        g = getattr(geocoder, provider)(addr_str)
        if g.latlng is None:
            time.sleep(.5)
            lats_lons.extend(lat_longs_from_geo_strings([addr_str], provider))
        else:
            lats_lons.append(g.latlng)
    return lats_lons
