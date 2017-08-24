import pandas as pd
import numpy as np
import os

"""
Helper functions to load the article data. The main method to use
is load_data().
"""

# Caution! Modifying this in code will have no effect since the
# default arguments are populated with this reference at creation
# time, so post-hoc modifications will do nothing.
__data_folder = os.path.join(os.path.split(__file__)[0], '..', 'data')

def load_articles(data_folder=__data_folder, nrows=None):
    """
    Loads the articles CSV. Can optionally only load the first
    `nrows` number of rows.
    """
    column_names = ['id',
                    'unknown1',
                    'url',
                    'orig_html',
                    'title',
                    'bodytext',
                    'relevant',
                    'created',
                    'last_modified',
                    'unknown2',
                    'feedname']

    return pd.read_csv(os.path.join(data_folder,
                                    'newsarticles_article.csv'),
                       header=None,
                       names=column_names,
                       nrows=nrows)


def load_categorizations(data_folder=__data_folder):
    """Loads the categorizations of the articles."""
    column_names = ['id', 'article_id', 'category_id']

    return pd.read_csv(os.path.join(data_folder,
                                    'newsarticles_article_categories.csv'),
                       header=None,
                       names=column_names)


def load_categories(data_folder=__data_folder):
    """Loads the mapping of id to names/abbrevations of categories"""
    column_names = ['id', 'category_name', 'abbreviation', 'created',
                    'unknown', 'group']

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

    tags_df = load_categorizations(data_folder)
    # will help cacheing
    tags_df.sort_values(by='article_id', inplace=True)

    categories_df = load_categories(data_folder)
    categories_df.set_index('id', drop=True, inplace=True)

    # tags_df['category_id'] = tags_df['category_id'].astype(str)
    tags_df['category_abbreviation'] = (categories_df
                                        ['abbreviation']
                                        [tags_df['category_id']]
                                        .values)

    if np.setdiff1d(tags_df['article_id'].values, df.index.values).size:
        print('Warning, tags were found for article IDs that do not exist.')

    article_ids = tags_df['article_id'].values
    cat_abbreviations = tags_df['category_abbreviation'].values

    # for some reason, some articles that are tagged don't show up
    # in the articles CSV. filter those out.
    existing_ids_filter = np.isin(article_ids, df.index.values)

    article_ids = article_ids[existing_ids_filter]
    cat_abbreviations = cat_abbreviations[existing_ids_filter]

    for i in range(tags_df['category_id'].max()):
        cat_name = categories_df.loc[i+1, 'abbreviation']
        df[cat_name] = 0
        df[cat_name] = df[cat_name].astype('int8') # save on that memory!
        df.loc[article_ids[cat_abbreviations == cat_name], cat_name] = 1

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
                     + ' '
)
    community_areas = pd.read_csv(os.path.join(data_folder, 'CommAreas.csv'))
    community_areas.set_index('AREA_NUM_1', inplace=True, drop=True)
    crime_string += (community_areas.loc[crimes['Community Area'], 'COMMUNITY']
                     .fillna('')
                     .values
                     + ' ')

    return crimes, crime_string
