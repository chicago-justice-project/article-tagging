# EXAMPLE
# https://maps.googleapis.com/maps/api/staticmap?size=400x400&markers=41.8850800,-87.6241350|41.880633,-87.629656&key=KEY

import webbrowser


def generate_api_string(lats_lons, key, size=400):
    print('Found {} addresses.'.format(len(lats_lons)))
    markers = []
    for addr in lats_lons:
        loc = '{},{}'.format(addr[0], addr[1])
        markers.append(loc)
    url_markers = '|'.join(markers)
    full_str = ('https://maps.googleapis.com/maps/api/staticmap'
                '?size={}x{}&markers={}&key={}').format(
        size, size, url_markers, key
    )
    return full_str


def url_open(url):
    webbrowser.open_new_tab(url)
