[![Build Status](https://travis-ci.org/chicago-justice-project/article-tagging.svg?branch=master)](https://travis-ci.org/chicago-justice-project/article-tagging)

Automatically classify news articles with type-of-crime tags? Neat! Also automatically extract location strings from the article? Even cooler!

```python
>>> import tagnews
>>> crimetags = tagnews.CrimeTags()
>>> article_text = ('The homicide occurred at the 1700 block of S. Halsted Ave.'
...   ' It happened just after midnight. Another person was killed at the'
...   ' intersection of 55th and Woodlawn, where a lone gunman')
>>> crimetags.tagtext_proba(article_text)
HOMI     0.739159
VIOL     0.146943
GUNV     0.134798
...
>>> geoextractor = tagnews.GeoCoder()
>>> geostrings = geoextractor.extract_geostrings(article_text, prob_thresh=0.5)
>>> geostrings
[['1700', 'block', 'of', 'S.', 'Halsted', 'Ave.'], ['55th', 'and', 'Woodlawn,']]
>>> lat_longs, _, _ = geoextractor.lat_longs_from_geostring_lists(geostrings)
>>> lat_longs
[[41.49612808227539, -87.63743591308594], [41.79513222479058, -87.58843505219843]]
```

# Installation

This library can be installed with pip:

```
pip install tagnews
```

You will need to install some [NLTK](http://www.nltk.org/) packages as well:

```python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('wordnet')
```

`tagnews` requires python >= 3.5.

# Sample Usage

The main classes are `tagnews.CrimeTags` and `tagnews.GeoCoder`.

```python
>>> import tagnews
>>> crimetags = tagnews.CrimeTags()
>>> article_text = ('The homicide occurred at the 1700 block of S. Halsted Ave.'
...   ' It happened just after midnight. Another person was killed at the'
...   ' intersection of 55th and Woodlawn, where a lone gunman')
>>> crimetags.tagtext_proba(article_text)
HOMI     0.739159
VIOL     0.146943
GUNV     0.134798
...
>>> crimetags.tagtext(article_text, prob_thresh=0.5)
['HOMI']
>>> geoextractor = tagnews.GeoCoder()
>>> prob_out = geoextractor.extract_geostring_probs(article_text)
>>> list(zip(*prob_out))
[..., ('at', 0.0044685714), ('the', 0.005466637), ('1700', 0.7173856),
 ('block', 0.81395197), ('of', 0.82227415), ('S.', 0.7940061),
 ('Halsted', 0.70529455), ('Ave.', 0.60538065), ...]
>>> geostrings = geoextractor.extract_geostrings(article_text, prob_thresh=0.5)
>>> geostrings
[['1700', 'block', 'of', 'S.', 'Halsted', 'Ave.'], ['55th', 'and', 'Woodlawn,']]
>>> lat_longs, scores, num_found = geoextractor.lat_longs_from_geostring_lists(geostrings)
>>> lat_longs
[[41.49612808227539, -87.63743591308594], [41.79513222479058, -87.58843505219843]]
>>> scores # our best attempt at giving a confidence in the lat_longs, higher is better
array([0.5913217, 0.       ], dtype=float32)
>>> num_found # how many results gisgraphy found for the (post-processed) geostring
[8, 10]
```

# How can I contribute?

Great question! Please see [CONTRIBUTING.md](https://github.com/chicago-justice-project/article-tagging/blob/master/CONTRIBUTING.md).

# Problems?

If you have problems, please [report an issue](https://github.com/chicago-justice-project/article-tagging/issues/new).

# Background

We want to compare the amount different types of crimes are reported in certain areas vs. the actual occurrence amount in those areas. Are some crimes under-represented in certain areas but over-represented in others? To accomplish this, we'll need to be able to extract a type-of-crime tag and geospatial data from news articles.


The [Chicago Justice Project](http://chicagojustice.org/) has been scraping RSS feeds of articles written by Chicago area news outlets for several years, allowing them to collect over 400,000 articles. At the same time, an amazing group of [volunteers](http://chicagojustice.org/volunteer-for-cjp/) have helped them tag these articles. The tags include crime categories like "Gun Violence", "Drugs", "Sexual Assault", but also organizations such as "Cook County State's Attorney's Office", "Illinois State Police", "Chicago Police Department", and other miscellaneous categories such as "LGBTQ", "Immigration". The volunteer UI was also recently updated to allow highlighting of geographic information.

A group actively working on this project meets every Tuesday at [Chi Hack Night](https://chihacknight.org/).

# See Also

* [Chicago Justice Project](http://chicagojustice.org/)
* [Other CJP coding projects](https://github.com/chicago-justice-project)
* [... including the database/web scraping side of things](https://github.com/chicago-justice-project/chicago-justice)
* [What is Chi Hack Night?](https://chihacknight.org/about.html)
