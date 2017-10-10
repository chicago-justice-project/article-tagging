from geopy.geocoders import GoogleV3
import time
import progressbar

def get_loc_list(dframe, test=False):
    ''' Search through dframe and collect articles that don't yet have lat/long coordinates. Only return articles
        with total number of addresses less than or equal to 2500. If test is True only return 60 articles. '''
    
    # Slice all lists from dframe.locations that have locations data
    
    ll_list = list(dframe.locations[dframe.locations.apply(len) > 0].values)
    
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    
    if test:
        total = 30
    else:
        total = 2500
        
    ll_list_2 = []
    for i in ll_list:
        if count2 < total:
            if 'lat_long' not in i[0].keys():
                ll_list_2.append(i)
                count1 += len(i)
                count2 += 1
   #     count5 += 1
   # 
   # print(len(ll_list), count3, count4, count5)
   # print('List has {} articles and {} addresses.'.format(count2, count1))
   # 
   # count = 0
   # for i in ll_list:
   #     count += len(i)
   # print('For test mode, {} articles with {} addresses will be processed.'.format(len(ll_list),count))
   # 
    return ll_list_2

def get_lat_long(dframe, api_key, test=False):
    ''' Find all addresses associated with each news article in the dataframe and return the list of dictionaries
        for the address info with the lat/long coords added as a key/value pair to it's associated dict. '''
    
    g = GoogleV3(api_key = api_key, timeout = 10)
    
    loc_list = get_loc_list(dframe, test=test)
    
    test_batch = 30  # For debug
    batch1 = 50      # Google's geocoder service per second rate limit is 50 qps
    batch2 = 2500    # Google's geocoder service daily rate limit is 2500 qpd
    
    count = 0
    for i in loc_list:
        count += len(i)
    
    count1 = 0       # batch1 counter
    count2 = 0       # batch2 counter
    location = []
    
    if test:
        count = 0
        for i in loc_list:
            count += len(i)
        print('For test mode, {} articles with {} addresses will be processed.'.format(len(loc_list),count))
        max_value = test_batch
    elif count < batch2:
        print('{} articles with {} addresses will be processed.'.format(len(loc_list),count))
        max_value = count
    else:
        count = 0
        for i in loc_list:
            count += len(i)
        print('{} articles with {} addresses will be processed.'.format(len(loc_list),count))
        max_value = batch2
        
    with progressbar.ProgressBar(max_value=max_value) as bar:
        for i,l in enumerate(loc_list):
            for j,m in enumerate(l):
                if count1 >= batch1:
                    count1 = 0
                    time.sleep(0.5)
                    #break
                else:
                    addr = m.get('cleaned text').rstrip('?,.:; ')
                    location = g.geocode(addr, components={'locality':'Chicago'})
                    loc_list[i][j]['lat_long'] = location
                    count1 += 1
            count2 += 1
            time.sleep(0.1)
            bar.update(count2)
                
    return loc_list