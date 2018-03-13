import re
from nltk.corpus import stopwords
# post processing function for geo strings extracted from geoextractor.extract_geostrings()
# right now it
#  - adds chicago if not present
#  - adds illinois if not present
# future ideas
# - zip codes: could check for numbers at the end of the string. could ensure that
#              said numbers are five digits long (or four dash five digits)


def post_process(addr_list):
    assert type(
        addr_list) == list, "address is not list. use outputs from geoextractor.\
        extract_geostrings(article_text)[i]"
    addr_list = [w for w in addr_list if w not in stopwords.words('english')]
    if 'chicago' not in [word.lower() for word in addr_list]:
        addr_list.append('Chicago')
    if not contains_il_str(addr_list):
            addr_list.append(', IL')
    ' '.join(addr_list)
    clean = re.sub('[\W]+ ', '', addr_string)
    return clean

def contains_il_str(address_list):
    for il_str in ['illinois', 'il']:
        if il_str in [word.lower() for word in addr_list]:
            return True
    return False
