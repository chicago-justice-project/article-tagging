import pandas as pd
import numpy as np
import re
import json
import os
import warnings

"""
Helper functions to load the article data. The main method to use
is load_data().
"""

# Caution! Modifying this in code will have no effect since the
# default arguments are populated with this reference at creation
# time, so post-hoc modifications will do nothing.
__data_folder = os.path.join(os.path.split(__file__)[0], '..', 'data')


def clean_string(s):
    """
    Clean all the HTML/Unicode nastiness out of a string.
    Replaces newlines with spaces.
    """

    return s.replace('\r', '').replace('\n', ' ').replace('\xa0', ' ').strip()


def load_articles(data_folder=__data_folder, nrows=None):
    """
    Loads the articles CSV. Can optionally only load the first
    `nrows` number of rows.
    """
    column_names = ['id',
                    'feedname',
                    'url',
                    'orig_html',
                    'title',
                    'bodytext',
                    'relevant',
                    'created',
                    'last_modified',
                    'news_source_id',
                    'author']

    return pd.read_csv(os.path.join(data_folder,
                                    'newsarticles_article.csv'),
                       header=None,
                       names=column_names,
                       nrows=nrows,
                       dtype={'orig_html': str, 'author': str})


def load_taggings(data_folder=__data_folder):
    """Loads the type-of-crime human tagging of the articles."""
    uc_column_names = ['id', 'date', 'relevant',
                       'article_id', 'user_id', 'locations']

    uc = pd.read_csv(os.path.join(data_folder,
                                  'newsarticles_usercoding.csv'),
                     header=None,
                     names=uc_column_names)

    uc.set_index('id', drop=True, inplace=True)

    uc_tags_column_names = ['id', 'usercoding_id', 'category_id']

    uc_tags = pd.read_csv(os.path.join(data_folder,
                                       'newsarticles_usercoding_categories.csv'),
                          header=None,
                          names=uc_tags_column_names)
    uc_tags.set_index('usercoding_id', drop=True, inplace=True)

    uc_tags['article_id'] = uc.loc[uc_tags.index, 'article_id']
    return uc_tags


def load_locations(data_folder=__data_folder):
    """Load the human-extracted locations from the articles."""
    uc_column_names = ['id', 'date', 'relevant',
                       'article_id', 'user_id', 'locations']

    uc = pd.read_csv(os.path.join(data_folder,
                                  'newsarticles_usercoding.csv'),
                     header=None,
                     names=uc_column_names)

    uc['locations'] = uc['locations'].apply(lambda x: json.loads(x))

    return uc


def load_categories(data_folder=__data_folder):
    """Loads the mapping of id to names/abbrevations of categories"""
    column_names = ['id', 'category_name', 'abbreviation', 'created',
                    'active', 'kind']

    return pd.read_csv(os.path.join(data_folder, 'newsarticles_category.csv'),
                       header=None,
                       names=column_names)


def load_data(data_folder=__data_folder, nrows=None):
    """
    Creates a dataframe of the article information and k-hot encodes the tags
    into columns called cat_NUMBER. The k-hot encoding is done assuming that the
    categories are 1-indexed and there are as many categories as the maximum
    value of the numerical cateogry_id column.

    Inputs:
        data_folder:
            A folder containing the data files in CSV format.
        nrows:
            Number of articles to load. Defaults to all, which uses about 4
            GB of memory.
    """

    df = load_articles(data_folder=data_folder, nrows=nrows)
    df['relevant'] = df['relevant'] == 't'
    df.rename(columns={'id': 'article_id'}, inplace=True)
    df.set_index('article_id', drop=True, inplace=True)
    # hopefully this will save some memory/space, can add back if needed
    del(df['orig_html'])

    tags_df = load_taggings(data_folder)
    # will help cacheing
    tags_df.sort_values(by='article_id', inplace=True)
    tags_df = tags_df.loc[tags_df['article_id'].isin(df.index.intersection(tags_df['article_id']))]

    locs_df = load_locations(data_folder)
    locs_df.sort_values(by='article_id', inplace=True)
    locs_df = locs_df.loc[locs_df['article_id'].isin(df.index.intersection(locs_df['article_id']))]

    # init with empty lists
    df['locations'] = np.empty([df.shape[0], 0]).tolist()
    df.loc[locs_df['article_id'].values, 'locations'] = locs_df['locations'].values

    def find_loc_in_string(locs, string):
        """
        The locations are generated from JavaScript, which means there's
        going to be some problems getting things to line up exactly and
        neatly. This function will hopefully performa all necessary
        transformations to find the given location text within the
        larger string.

        Inputs:
            locs: list of locations as loaded by load_locations
            string: bodytext of article in which to find locs
        Returns:
            updated_locs: list of locations as loaded by
                load_locations, but with a couple
                extra fields ('cleaned text' and 'cleaned span').
        """

        for i, loc in enumerate(locs):
            loc_text = loc['text']

            loc_text = clean_string(loc_text)
            string = clean_string(string)

            loc['cleaned text'] = loc_text

            spans = [x.span() for x in re.finditer(re.escape(loc_text), string)]
            if spans:
                # The string may have occurred multiple times, and since the spans
                # don't line up perfectly we can't know which one is the "correct" one.
                # Best we can do is find the python span closest to the expected
                # javascript span.
                closest = np.abs(np.argmin(np.array([x[0] for x in spans]) - loc['start']))
                loc['cleaned span'] = spans[closest]

            locs[i] = loc

        return locs

    df['locations'] = df.apply(
        lambda r: find_loc_in_string(r['locations'], r['bodytext']),
        axis=1
    )

    num_no_match = df['locations'].apply(
        lambda locs: any([('cleaned span' not in loc) for loc in locs])
    ).sum()
    if num_no_match:
        warnings.warn(str(num_no_match) + ' location strings were not found in the bodytext.',
                      RuntimeWarning)

    categories_df = load_categories(data_folder)
    categories_df.set_index('id', drop=True, inplace=True)

    # tags_df['category_id'] = tags_df['category_id'].astype(str)
    tags_df['category_abbreviation'] = (categories_df
                                        ['abbreviation']
                                        [tags_df['category_id']]
                                        .values)

    if np.setdiff1d(tags_df['article_id'].values, df.index.values).size:
        warnings.warn('Tags were found for article IDs that do not exist.',
                      RuntimeWarning)

    article_ids = tags_df['article_id'].values
    cat_abbreviations = tags_df['category_abbreviation'].values

    # for some reason, some articles that are tagged don't show up
    # in the articles CSV. filter those out.
    existing_ids_filter = np.isin(article_ids, df.index.values)

    article_ids = article_ids[existing_ids_filter]
    cat_abbreviations = cat_abbreviations[existing_ids_filter]

    for i in range(categories_df.shape[0]):
        cat_name = categories_df.loc[i+1, 'abbreviation']
        df[cat_name] = 0
        df[cat_name] = df[cat_name].astype('int8') # save on that memory!
        df.loc[article_ids[cat_abbreviations == cat_name], cat_name] = 1

    df.loc[df['bodytext'].isnull(), 'bodytext'] = ''

    return df


def load_crime_data(data_folder=__data_folder):
    crimes = pd.read_csv(os.path.join(data_folder, 'Crimes.csv'))
    crimes = crimes[crimes['Year'] > 2010]

    crime_string = pd.Series('', crimes.index)

    # ['ID', 'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type',
    #  'Description', 'Location Description', 'Arrest', 'Domestic', 'Beat',
    #  'District', 'Ward', 'Community Area', 'FBI Code', 'X Coordinate',
    #  'Y Coordinate', 'Year', 'Updated On', 'Latitude', 'Longitude',
    #  'Location']

    # TODO: synonyms on this for month name, weekday name,
    # time of day (e.g. afternoon), etc.
    crime_string += crimes['Date'] + ' '

    # TODO: synonyms?
    crime_string += crimes['Primary Type'] + ' '

    # TODO: synonyms?
    crime_string += crimes['Description'] + ' '

    # TODO: synonyms?
    crime_string += crimes['Location Description'] + ' '

    # TODO: synonyms?
    iucr = pd.read_csv(os.path.join(data_folder, 'IUCR.csv'))
    iucr.set_index('IUCR', drop=True, inplace=True)
    idx = iucr.index
    idx_values = idx.values
    idx_values[idx.str.len() == 3] = '0' + idx_values[idx.str.len() == 3]
    crime_string += (iucr.loc[crimes['IUCR'], 'PRIMARY DESCRIPTION']
                     .fillna('')
                     .values
                     + ' ')
    crime_string += (iucr.loc[crimes['IUCR'], 'SECONDARY DESCRIPTION']
                     .fillna('')
                     .values
                     + ' ')
    community_areas = pd.read_csv(os.path.join(data_folder, 'CommAreas.csv'))
    community_areas.set_index('AREA_NUM_1', inplace=True, drop=True)
    crime_string += (community_areas.loc[crimes['Community Area'], 'COMMUNITY']
                     .fillna('')
                     .values
                     + ' ')

    return crimes, crime_string


def load_ner_data(data_folder=__data_folder):
    """
    Loads ner.csv from the specified data folder.

    The column 'stag' is a binary value indicating whether or not
    the row corresponds to the entity "geo". Typically, you will
    want to use column 'word' to predict the column 'stag'.
    """
    df = pd.read_csv(os.path.join(data_folder, 'ner.csv'),
                     encoding="ISO-8859-1",
                     error_bad_lines=False,
                     index_col=0)

    df.dropna(subset=['word', 'tag'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['stag'] = (df['tag'] == 'B-geo') | (df['tag'] == 'I-geo')
    df['all_tags'] = df['tag']
    df['tag'] = df['stag']
    df = df[['word', 'all_tags', 'tag']]

    return df
