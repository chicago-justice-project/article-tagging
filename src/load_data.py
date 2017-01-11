import pandas as pd
import os

DATA_FOLDER = '../cjp-db/'

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
    df.rename(columns={'id': 'article_id'}, inplace=True)
    df.set_index('article_id', drop=True, inplace=True)
    # hopefully this will save some memory/space, can add back if needed
    del(df['orig_html'])

    tags_df = load_categorizations()
    # will help cacheing
    tags_df.sort_values(by='article_id', inplace=True)

    categories_df = load_categories()
    categories_df.set_index('id', drop=True, inplace=True)

    for i in range(tags_df['category_id'].max() - 1):
        # cat_name = 'cat_' + str(i+1)
        cat_name = categories_df.loc[i+1, 'abbreviation']
        df[cat_name] = 0
        df[cat_name] = df[cat_name].astype('int8') # save on that memory!

    # tags_df['category_id'] = tags_df['category_id'].astype(str)
    tags_df['category_id'] = categories_df['abbreviation'][tags_df['category_id']]
    for _, row in tags_df.iterrows():
        df.loc[row['article_id'], 'cat_' + row['category_id']] = 1

    return df
