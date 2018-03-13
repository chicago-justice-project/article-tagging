import re

# post processing function for geo strings extracted from geoextractor.extract_geostrings()
# right now it
#  - adds chicago if not present
#  - adds illinois if not present
# future ideas
# - nltk remove stop words: obviously these shouldn't come out of the rnn,
#                           but they might sometimes. it's a harmless failsafe
#                           and easy to add
# - zip codes: could check for numbers at the end of the string. could ensure that
#              said numbers are five digits long (or four dash five digits)


def post_process(addr_list):
    assert type(
        addr_list) == list, "address is not list. use outputs from geoextractor.\
        extract_geostrings(article_text)[i]"
    if 'chicago' not in [word.lower() for word in addr_list]:
        addr_list.append('Chicago')
    contains_il_str = False
    for il_str in ['illinois', 'il']:
        if il_str in [word.lower() for word in addr_list]:
            contains_il_str = True
        if not contains_il_str:
            addr_list.append(', IL')
    ' '.join(addr_list)
    clean = re.sub('[\W]+ ', '', addr_string)
    return clean
