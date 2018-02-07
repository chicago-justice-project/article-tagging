import geocoder
import time

def lat_longs_from_geo_strings(lst):
    lats_lons = []
    for addr_str in lst:
        g = geocoder.google(addr_str)
        if g.latlng is None:
            time.sleep(.5)
            return lat_longs_from_geo_strings([addr_str])
        else:
            lats_lons.append(g.latlng)
    return lats_lons
