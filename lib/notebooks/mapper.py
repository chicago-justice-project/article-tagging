
def mapper(loclist, api_key):
    baseURL = 'http://maps.googleapis.com/maps/api/staticmap?center=Chicago,IL&zoom=11&size=640x640&scale=2'
    
    markers = '&markers=size:mid&color:red'
    for i in loclist:
        for j in i:
            if 'lat_long' in j.keys():
                markers += '%7C' + str(j['lat_long'][1][0]) + ',' + str(j['lat_long'][1][1])
    
    return baseURL + markers + '&key=' + api_key