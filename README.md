[![Build Status](https://travis-ci.org/chicago-justice-project/article-tagging.svg?branch=master)](https://travis-ci.org/chicago-justice-project/article-tagging)

# tagnews

`tagnews` is a Python library that can

* Automatically categorize the text from news articles with type-of-crime tags, e.g. homicide, arson, gun violence, etc.
* Automatically extract the locations discussed in the news article text, e.g. "55th and Woodlawn" and "1700 block of S. Halsted".
* Retrieve the latitude/longitude pairs for said locations using an instance of the pelias geocoder hosted by CJP.
* Get the community areas those lat/long pairs belong to using a shape file downloaded from the city data portal parsed by the `shapely` python library.

Sound interesting? There's example usage below!

You can find the source code on [GitHub](https://github.com/chicago-justice-project/article-tagging).

## Installation

You can install `tagnews` with pip:

```
pip install tagnews
```

**NOTE:** You will need to install some [NLTK](http://www.nltk.org/) packages as well:

```python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('wordnet')
```

Beware, `tagnews` requires python >= 3.5.

## Example

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
>>> coords, scores = geoextractor.lat_longs_from_geostring_lists(geostrings)
>>> coords
         lat       long
0  41.859021 -87.646934
1  41.794816 -87.597422
>>> scores # confidence in the lat/longs as returned by pelias, higher is better
array([0.878, 1.   ])
>>> geoextractor.community_area_from_coords(coords)
['LOWER WEST SIDE', 'HYDE PARK']
```

## Limitations

This project uses Machine Learning to automate data cleaning/preparation tasks that would be cost and time prohibitive to perform using people. Like all Machine Learning projects, *the results are not perfect, and in some cases may look just plain bad*.

We strived to build the best models possible, but perfect accuracy is rarely possible. If you have thoughts on how to do better, please consider [reporting an issue](https://github.com/chicago-justice-project/article-tagging/issues/new), or better yet  [contributing](https://github.com/chicago-justice-project/article-tagging/blob/master/CONTRIBUTING.md).

## How can I contribute?

Great question! Please see [CONTRIBUTING.md](https://github.com/chicago-justice-project/article-tagging/blob/master/CONTRIBUTING.md).

## Problems?

If you have problems, please [report an issue](https://github.com/chicago-justice-project/article-tagging/issues/new). Anything that is behaving unexpectedly is an issue, and should be reported. If you are getting bad or unexpected results, that is also an issue, and should be reported. We may not be able to do anything about it, but more data rarely degrades performance.

## Background

We want to compare the amount of different types of crimes are reported in certain areas vs. the actual occurrence amount in those areas. In essence, *are some crimes under-represented in certain areas but over-represented in others?* This is the main question driving the analysis.

This question came from the [Chicago Justice Project](http://chicagojustice.org/). They have been interested in answering this question for quite a while, and have been collecting the data necessary to have a data-backed answer. Their efforts include

1. Scraping RSS feeds of articles written by Chicago area news outlets for several years, allowing them to collect almost half a million articles.
2. Organizing an amazing group of [volunteers](http://chicagojustice.org/volunteer-for-cjp/) that have helped them tag these articles with crime categories like "Gun Violence" and "Drugs", but also organizations such as "Cook County State's Attorney's Office", "Illinois State Police", "Chicago Police Department", and other miscellaneous categories such as "LGBTQ", "Immigration".
3. The web UI used to do this tagging was also recently updated to allow highlighting of geographic information, resulting in several hundred articles with labeled location sub-strings.

Most of the code for those components can be found [here](https://github.com/chicago-justice-project/chicago-justice).

A group actively working on this project meets every Tuesday at [Chi Hack Night](https://chihacknight.org/).

## See Also

* [Chicago Justice Project](http://chicagojustice.org/)
* [Source code of other CJP projects](https://github.com/chicago-justice-project)
* [... including the database/web scraping side of things](https://github.com/chicago-justice-project/chicago-justice)
* [What is Chi Hack Night?](https://chihacknight.org/about.html)
