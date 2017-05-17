import pandas as pd
import os

DATA_FOLDER = '../data/'

def load_articles():
    """Loads the articles CSV."""
    column_names = ['id',
                    'feedname',
                    'url',
                    'orig_html',
                    'title',
                    'bodytext',
                    'relevant',
                    'created',
                    'last_modified']

    return pd.read_csv(os.path.join(DATA_FOLDER,
                                    'newsarticles_article.csv'),
                       header=None,
                       names=column_names)


def load_categorizations():
    """Loads the categorizations of the articles."""
    column_names = ['id', 'article_id', 'category_id']

    return pd.read_csv(os.path.join(DATA_FOLDER,
                                    'newsarticles_article_categories.csv'),
                       header=None,
                       names=column_names)


def load_categories():
    """Loads the mapping of id to names/abbrevations of categories"""
    column_names = ['id', 'category_name', 'abbreviation', 'created']

    return pd.read_csv(os.path.join(DATA_FOLDER, 'newsarticles_category.csv'),
                       header=None,
                       names=column_names)


def load_data():
    """Creates a dataframe of the article information and k-hot encodes the tags
    into columns called cat_NUMBER. The k-hot encoding is done assuming that the
    categories are 1-indexed and there are as many categories as the maximum
    value of the numerical cateogry_id column."""

    df = load_articles()
    df['relevant'] = df['relevant'] == 't'
    df.rename(columns={'id': 'article_id'}, inplace=True)
    df.set_index('article_id', drop=True, inplace=True)
    # hopefully this will save some memory/space, can add back if needed
    del(df['orig_html'])

    tags_df = load_categorizations()
    # will help cacheing
    tags_df.sort_values(by='article_id', inplace=True)

    categories_df = load_categories()
    categories_df.set_index('id', drop=True, inplace=True)

    for i in range(tags_df['category_id'].max()):
        # cat_name = 'cat_' + str(i+1)
        cat_name = categories_df.loc[i+1, 'abbreviation']
        df[cat_name] = 0
        df[cat_name] = df[cat_name].astype('int8') # save on that memory!

    # tags_df['category_id'] = tags_df['category_id'].astype(str)
    tags_df['category_abbreviation'] = categories_df['abbreviation'][tags_df['category_id']].values
    for _, row in tags_df.iterrows():
        df.loc[row['article_id'], row['category_abbreviation']] = 1

    return df


def load_crime_data():
    crimes = pd.read_csv(os.path.join(DATA_FOLDER, 'Crimes.csv'))
    crimes = crimes[crimes['Year'] > 2010]

    crime_string = pd.Series('', crimes.index)

    # ['ID', 'Case Number', 'Date', 'Block', 'IUCR', 'Primary Type',
    #  'Description', 'Location Description', 'Arrest', 'Domestic', 'Beat',
    #  'District', 'Ward', 'Community Area', 'FBI Code', 'X Coordinate',
    #  'Y Coordinate', 'Year', 'Updated On', 'Latitude', 'Longitude',
    #  'Location']

    # TODO: synonyms on this for month name, weekday name, time of day (e.g. afternoon), etc.
    crime_string += crimes['Date'] + ' '

    # TODO: synonyms?
    crime_string += crimes['Primary Type'] + ' '

    # TODO: synonyms?
    crime_string += crimes['Description'] + ' '

    # TODO: synonyms?
    crime_string += crimes['Location Description'] + ' '

    # TODO: synonyms?
    iucr = pd.read_csv(os.path.join(DATA_FOLDER, 'IUCR.csv'))
    iucr.set_index('IUCR', drop=True, inplace=True)
    idx = iucr.index
    idx_values = idx.values
    idx_values[idx.str.len() == 3] = '0' + idx_values[idx.str.len() == 3]
    crime_string += iucr.loc[crimes['IUCR'], 'PRIMARY DESCRIPTION'].fillna('').values + ' '
    crime_string += iucr.loc[crimes['IUCR'], 'SECONDARY DESCRIPTION'].fillna('').values + ' '

    community_areas = pd.read_csv(os.path.join(DATA_FOLDER, 'CommAreas.csv'))
    community_areas.set_index('AREA_NUM_1', inplace=True, drop=True)
    crime_string += community_areas.loc[crimes['Community Area'], 'COMMUNITY'].fillna('').values + ' '

    return crimes, crime_string
