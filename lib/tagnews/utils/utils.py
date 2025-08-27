import os
import requests
import shutil
import pandas as pd
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_all_data():
    # fmt: off
    """ articles_df, categories_df, trainedcategoryrelevance_df, trainedcoding_df, usercoding_df, usercoding_categories_df, trainedlocation"""
    # fmt: on
    newssource = load_newssource()
    articles = load_articles()
    categories = load_categories()
    trainedcategoryrelevance = load_trainedcategoryrelevance()
    trainedlocation = load_trainedlocation()
    trainedcoding = load_trainedcoding()
    usercoding = load_usercoding()
    usercoding_categories = load_usercoding_categories()
#    trainedsentiment = load_trainedsentiment()
#    trainedsentimententities = load_trainedsentimententities()
    return (
        newssource,
        articles,
        categories,
        trainedcategoryrelevance,
        trainedcoding,
        usercoding,
        usercoding_categories,
        trainedlocation,
#        trainedsentiment,
#        trainedsentimententities
    )

def load_data_subset():
    # fmt: off
    """ articles_df, categories_df, trainedcategoryrelevance_df, trainedcoding_df, usercoding_df, usercoding_categories_df, trainedlocation"""
    # fmt: on
    newssource = load_newssource()
    articles = load_articles_nohtml()
    categories = load_categories()
    trainedcategoryrelevance = load_trainedcategoryrelevance()
    trainedlocation = load_trainedlocation()
    trainedcoding = load_trainedcoding()
    usercoding = load_usercoding()
    usercoding_categories = load_usercoding_categories()

    return (
        newssource,
        articles,
        categories,
        trainedcategoryrelevance,
        trainedcoding,
        usercoding,
        usercoding_categories,
        trainedlocation
    )

def load_newssource():
    newsource = pd.read_csv(
        "./cjp_tables/newsarticles_newssource.csv.gz", header=None, compression="gzip", low_memory=False
    )
    newsource.columns = [
        "source_id",
        "source_name",
        "short_name",
        "legacy_feed_id",
    ]
    print(f"news sources loaded. size: {newsource.shape}")
    return newsource


def load_articles():
    # Read CSV file of articles but exclude the original html (orig_html) column
    article = pd.read_csv(
        "./cjp_tables/newsarticles_article.csv.gz", header=None, usecols=[0,1,2,4,5,6,7,8,9,10], compression="gzip", low_memory=False
    )
    article.columns = [
        "id",
        "feedname",
        "url",
        "title",
        "bodytext",
        "relevant",
        "created",
        "last_modified",
        "news_source_id",
        "author",
    ]
    print(f"articles loaded. size: {article.shape}")
    return article

def load_articles_nohtml():
    article = pd.read_csv(
        "./cjp_tables/newsarticles_article.csv.gz", header=None, compression="gzip", low_memory=False
    )
    article.columns = [
        "id",
        "feedname",
        "url",
        "title",
        "bodytext",
        "relevant",
        "created",
        "last_modified",
        "news_source_id",
        "author",
    ]
    print(f"articles loaded. size: {article.shape}")
    return article


def load_categories():
    categories = pd.read_csv(
        "./cjp_tables/newsarticles_category.csv.gz", header=None, compression="gzip", low_memory=False
    )
    categories.columns = ["id", "title", "abbreviation", "created", "active", "kind"]
    print(f"categories loaded. size: {categories.shape}")
    return categories


def load_trainedcategoryrelevance():
    trainedcategoryrelevance = pd.read_csv(
        "./cjp_tables/newsarticles_trainedcategoryrelevance.csv.gz", header=None, compression="gzip", low_memory=False
    )
    trainedcategoryrelevance.columns = ["id", "relevance", "category_id", "coding_id"]
    print(f"trainedcategoryrelevance loaded. size: {trainedcategoryrelevance.shape}")
    return trainedcategoryrelevance


def load_trainedcoding():
    trainedcoding = pd.read_csv(
        "./cjp_tables/newsarticles_trainedcoding.csv.gz",
        header=None,
        compression="gzip",
        low_memory=False
    )
    trainedcoding.columns = [
        "id",
        "date",
        "model_info",
        "relevance",
        "article_id",
        "sentiment",
        "bin",
        "sentiment_processed",
    ]
    print(f"trainedcoding loaded. size: {trainedcoding.shape}")
    return trainedcoding


def load_trainedlocation():
    trainedlocation = pd.read_csv(
        "./cjp_tables/newsarticles_trainedlocation.csv.gz",
        header=None,
        compression="gzip",
        low_memory=False
    )
    trainedlocation.columns = [
        "id",
        "text",
        "latitude",
        "longitude",
        "coding_id",
        "confidence",
        "neighborhood",
        "is_best"
    ]
    print(f"trainedlocation loaded. size: {trainedlocation.shape}")
    return trainedlocation


def load_usercoding():
    usercoding = pd.read_csv(
        "./cjp_tables/newsarticles_usercoding.csv.gz", header=None, compression="gzip", low_memory=False
    )
    usercoding.columns = [
        "id",
        "date",
        "relevant",
        "article_id",
        "user_id",
        "locations",
        "sentiment",
    ]
    print(f"usercoding loaded. size: {usercoding.shape}")
    return usercoding


def load_usercoding_categories():
    usercoding_categories = pd.read_csv(
        "./cjp_tables/newsarticles_usercoding_categories.csv.gz",
        header=None,
        compression="gzip",
        low_memory=False
    )
    usercoding_categories.columns = ["id", "usercoding_id", "category_id"]
    print(f"usercoding_categories loaded. size: {usercoding_categories.shape}")
    return usercoding_categories


def load_trainedsentiment():
    trainedsentiment = pd.read_csv(
        "./cjp_tables/newsarticles_trainedsentiment.csv.gz",
        header=None,
        compression="gzip",
        low_memory=False
    )
    trainedsentiment.columns = [
        "id",
        "date",
        "api_response",
        "coding_id",
    ]
    print(f"trainedsentiment loaded. size: {trainedsentiment.shape}")
    return trainedsentiment


def load_trainedsentimententities():
    trainedsentimententities = pd.read_csv(
        "./cjp_tables/newsarticles_trainedsentimententities.csv.gz",
        header=None,
        compression="gzip",
        low_memory=False
    )
    trainedsentimententities.columns = [
        "id",
        "index",
        "entity",
        "sentiment",
        "coding_id",
        "response_id",
    ]
    print(f"trainedsentimententities loaded. size: {trainedsentimententities.shape}")
    return trainedsentimententities
