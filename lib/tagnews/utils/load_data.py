import pandas as pd
import numpy as np
import re
import json
import os
import warnings
import shutil
from pathlib import Path
import codecs

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
    uc_column_names = ['id', 'date', 'relevant', 'article_id',
                       'user_id', 'locations', 'sentiment']

    uc = pd.read_csv(os.path.join(data_folder,
                                  'newsarticles_usercoding.csv'),
                     header=None,
                     names=uc_column_names)

    uc.set_index('id', drop=True, inplace=True)

    uc_tags_column_names = ['id', 'usercoding_id', 'category_id']

    uc_tags = pd.read_csv(
        os.path.join(data_folder, 'newsarticles_usercoding_categories.csv'),
        header=None,
        names=uc_tags_column_names
    )
    uc_tags.set_index('usercoding_id', drop=True, inplace=True)

    uc_tags['article_id'] = uc.loc[uc_tags.index, 'article_id']
    return uc_tags


def load_model_categories(data_folder=__data_folder):
    tcr_names = ['id', 'relevance', 'category_id', 'coding_id']
    tc_names = ['id', 'date', 'model_info', 'relevance', 'article_id',
                'sentiment']

    tcr = pd.read_csv(
        os.path.join(data_folder, 'newsarticles_trainedcategoryrelevance.csv'),
        names=tcr_names
    )
    tc = pd.read_csv(
        os.path.join(data_folder, 'newsarticles_trainedcoding.csv'),
        names=tc_names
    ).set_index('id', drop=True)
    tcr['article_id'] = tc.loc[tcr['coding_id']]['article_id'].values
    return tcr


def load_model_locations(data_folder=__data_folder):
    tl_names = ['id', 'text', 'latitude', 'longitude', 'coding_id',
                'confidence', 'neighborhood']
    tc_names = ['id', 'date', 'model_info', 'relevance', 'article_id',
                'sentiment']

    tl = pd.read_csv(
        os.path.join(data_folder, 'newsarticles_trainedlocation.csv'),
        names=tl_names
    )
    tc = pd.read_csv(
        os.path.join(data_folder, 'newsarticles_trainedcoding.csv'),
        names=tc_names
    ).set_index('id', drop=True)
    tl['article_id'] = tc.loc[tl['coding_id']]['article_id'].values
    return tl


def load_locations(data_folder=__data_folder):
    """Load the human-extracted locations from the articles."""
    uc_column_names = ['id', 'date', 'relevant', 'article_id',
                       'user_id', 'locations', 'sentiment']

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
    into columns called cat_NUMBER. The k-hot encoding is done assuming that
    the categories are 1-indexed and there are as many categories as the
    maximum value of the numerical cateogry_id column.
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
    tags_df = tags_df.loc[tags_df['article_id'].isin(
        df.index.intersection(tags_df['article_id']))]

    locs_df = load_locations(data_folder)
    locs_df.sort_values(by='article_id', inplace=True)
    locs_df = locs_df.loc[locs_df['article_id'].isin(
        df.index.intersection(locs_df['article_id']))]

    model_tags_df = load_model_categories(data_folder)
    # will help cacheing
    model_tags_df.sort_values(by='article_id', inplace=True)
    model_tags_df = model_tags_df.loc[model_tags_df['article_id'].isin(
        df.index.intersection(model_tags_df['article_id']))]

    # init with empty lists
    df['locations'] = np.empty([df.shape[0], 0]).tolist()
    loc_article_ids = locs_df['article_id'].values
    df.loc[loc_article_ids, 'locations'] = locs_df['locations'].values

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
            loc_txt = loc['text']

            loc_txt = clean_string(loc_txt)
            string = clean_string(string)

            loc['cleaned text'] = loc_txt

            spans = [x.span() for x in re.finditer(re.escape(loc_txt), string)]
            if spans:
                # The string may have occurred multiple times, and since the
                # spans don't line up perfectly we can't know which one is the
                # "correct" one. Best we can do is find the python span closest
                # to the expected javascript span.
                closest = np.argmin(np.abs(
                    np.array([x[0] for x in spans]) - loc['start']
                ))
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
        warnings.warn(('{} location strings were not found in'
                       ' the bodytext.').format(num_no_match),
                      RuntimeWarning)

    model_locations_df = load_model_locations(data_folder)
    model_locations_df = model_locations_df.set_index('article_id')
    model_locations_gb = model_locations_df.groupby('article_id')
    model_locations_text = model_locations_gb['text'].apply(list)
    df['model_location_text'] = model_locations_text

    categories_df = load_categories(data_folder)
    categories_df.set_index('id', drop=True, inplace=True)

    # tags_df['category_id'] = tags_df['category_id'].astype(str)
    tags_df['category_abbreviation'] = (categories_df
                                        ['abbreviation']
                                        [tags_df['category_id']]
                                        .values)
    model_tags_df['category_abbreviation'] = (categories_df
                                              ['abbreviation']
                                              [model_tags_df['category_id']]
                                              .values)

    if np.setdiff1d(tags_df['article_id'].values, df.index.values).size:
        warnings.warn('Tags were found for article IDs that do not exist.',
                      RuntimeWarning)

    def update_df_with_categories(article_ids, cat_abbreviations, vals,
                                  is_model):
        # for some reason, some articles that are tagged don't show up
        # in the articles CSV. filter those out.
        existing_ids_filter = np.isin(article_ids, df.index.values)

        article_ids = article_ids[existing_ids_filter]
        cat_abbreviations = cat_abbreviations[existing_ids_filter]
        vals = vals[existing_ids_filter]

        for i in range(categories_df.shape[0]):
            cat_name = categories_df.loc[i+1, 'abbreviation']
            if is_model:
                cat_name += '_model'
            df[cat_name] = 0
            if not is_model:
                df[cat_name] = df[cat_name].astype('int8')
            matches = cat_abbreviations == cat_name
            if not matches.sum():
                continue
            df.loc[article_ids[matches], cat_name] = vals[matches]

    update_df_with_categories(
        model_tags_df['article_id'].values,
        model_tags_df['category_abbreviation'].values + '_model',
        model_tags_df['relevance'].values,
        is_model=True
    )
    update_df_with_categories(
        tags_df['article_id'].values,
        tags_df['category_abbreviation'].values,
        np.ones((tags_df['article_id'].values.shape), dtype='int8'),
        is_model=False
    )

    df.loc[df['bodytext'].isnull(), 'bodytext'] = ''

    return df


def subsample_and_resave(out_folder, n=5, input_folder=__data_folder,
                         random_seed=5):
    """
    Subsamples the CSV data files so that we have at least
    `n` articles from each type-of-crime tag as determined
    by the human coding. Saves the subsampled CSV data
    into `out_folder`. If there are fewer than `n` articles
    tagged with a type-of-crime, then we will use all of
    the articles with that tag.
    Inputs
    ------
    out_folder : str
        Path to folder where data should be saved. Should already exist.
    n : int
        How many examples from each category should we have?
    input_folder : str
        Path to where the full CSV files are saved.
    random_seed : None or int
        np.random.RandomState() will be seeded with this value
        in order to perform the random subsampling.
    """
    out_folder = str(Path(out_folder).expanduser().absolute())
    input_folder = str(Path(input_folder).expanduser().absolute())
    if out_folder == input_folder:
        raise RuntimeError('out_folder cannot match input_folder.')

    random_state = np.random.RandomState(random_seed)

    df = load_data(input_folder)
    chosen_indexes = []
    for crime_type in df.loc[:, 'OEMC':].columns:
        is_type = df[crime_type].astype(bool)
        n_samps = min(n, is_type.sum())
        chosen_indexes += (df.loc[is_type, :]
                           .sample(n_samps, random_state=random_state)
                           .index
                           .tolist())
    del df

    chosen_indexes = sorted(list(set(chosen_indexes)))

    # newsarticles_article.csv
    articles_df = load_articles(input_folder)
    sample = (articles_df
              .reset_index()
              .set_index('id')
              .loc[chosen_indexes, 'index'])
    articles_df = articles_df.loc[sample, :]
    # garble garble
    articles_df['bodytext'] = articles_df['bodytext'].astype(str).apply(
        lambda x: codecs.encode(x, 'rot-13')
    )
    articles_df.to_csv(os.path.join(out_folder, 'newsarticles_article.csv'),
                       header=None, index=False)
    del articles_df

    # newsarticles_category.csv
    shutil.copyfile(os.path.join(input_folder, 'newsarticles_category.csv'),
                    os.path.join(out_folder, 'newsarticles_category.csv'))

    # newsarticles_usercoding.csv
    uc_column_names = ['id', 'date', 'relevant',
                       'article_id', 'user_id', 'locations']

    uc_df = pd.read_csv(os.path.join(input_folder,
                                     'newsarticles_usercoding.csv'),
                        header=None,
                        names=uc_column_names)

    sample = np.where(uc_df['article_id'].isin(chosen_indexes))[0]
    uc_df.loc[sample, :].to_csv(
        os.path.join(out_folder, 'newsarticles_usercoding.csv'),
        header=None, index=False
    )

    uc_tags_column_names = ['id', 'usercoding_id', 'category_id']

    # newsarticles_usercoding_categories.csv
    uc_tags_df = pd.read_csv(
        os.path.join(input_folder,
                     'newsarticles_usercoding_categories.csv'),
        header=None,
        names=uc_tags_column_names,
        dtype={'id': int, 'usercoding_id': int, 'category_id': int}
    )
    sample = np.where(uc_df
                      .set_index('id')
                      .loc[uc_tags_df['usercoding_id'], 'article_id']
                      .isin(chosen_indexes)
                      )[0]
    uc_tags_df = uc_tags_df.loc[sample, :]
    uc_tags_df.to_csv(
        os.path.join(out_folder, 'newsarticles_usercoding_categories.csv'),
        header=None, index=False
    )

    # newsarticles_trainedcoding
    tc_names = ['id', 'date', 'model_info', 'relevance', 'article_id']
    tc = pd.read_csv(
        'tagnews/data/newsarticles_trainedcoding.csv',
        names=tc_names
    )
    tc = tc.loc[tc['article_id'].isin(chosen_indexes)]
    tc.to_csv(
        os.path.join(out_folder, 'newsarticles_trainedcoding.csv'),
        header=False, index=False
    )

    # newsarticles_trainedcategoryrelevance
    tcr_names = ['id', 'relevance', 'category_id', 'coding_id']
    tcr = pd.read_csv(
        'tagnews/data/newsarticles_trainedcategoryrelevance.csv',
        names=tcr_names
    )
    tcr = tcr.loc[tcr['coding_id'].isin(tc['id'])]
    tcr.to_csv(
        os.path.join(out_folder, 'newsarticles_trainedcategoryrelevance.csv'),
        header=False, index=False
    )

    # newsarticles_trainedlocation
    tl_names = ['id', 'text', 'latitude', 'longitude', 'coding_id']
    tl = pd.read_csv(
        'tagnews/data/newsarticles_trainedlocation.csv',
        names=tl_names
    )
    tl = tl.loc[tl['coding_id'].isin(tc['id'])]
    tl.to_csv(
        os.path.join(out_folder, 'newsarticles_trainedlocation.csv'),
        header=False, index=False
    )


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