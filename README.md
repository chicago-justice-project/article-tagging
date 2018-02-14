Automatically classify news articles with type-of-crime tags? Neat! Also automatically extract location strings from the article? Even cooler!

```python
>>> import tagnews
>>> crimetags = tagnews.CrimeTags()
>>> article_text = 'The homicide occurred at the 1700 block of S. Halsted Ave. It happened just after midnight. Another person was killed at the intersection of 55th and Woodlawn, where a lone gunman'
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
[..., ('at', 0.0044685714), ('the', 0.005466637), ('1700', 0.7173856), ('block', 0.81395197), ('of', 0.82227415), ('S.', 0.7940061), ('Halsted', 0.70529455), ('Ave.', 0.60538065), ...]
>>> geoextractor.extract_geostrings(article_text, prob_thresh=0.5)
[['1700', 'block', 'of', 'S.', 'Halsted', 'Ave.'], ['55th', 'and', 'Woodlawn,']]
>>> import os; import psutil
>>> print('Memory usage: {} MB'.format(psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)))
Memory usage: 524.7265625 MB
```

***The documentation for this project is a work in progress. If something is unclear, or worse yet, incorrect, please report that as an [issue](https://github.com/chicago-justice-project/article-tagging/issues).***

# Installation

If you are wanting to install this to use as a package that can deliver NLP results out of the box, then please see [INSTALLATION.md](INSTALLATION.md). If you are wanting to roll up your sleeves and do some data science, please see [CONTRIBUTING.md](CONTRIBUTING.md).

# Usage

Below are sample usages when you want to just use this as a library to make predictions.

## From python

The main classes are `tagnews.CrimeTags` and `tagnews.GeoCoder`:

```python
>>> import tagnews
>>> crimetags = tagnews.CrimeTags()
>>> article_text = 'The homicide occurred at the 1700 block of S. Halsted Ave. It happened just after midnight. Another person was killed at the intersection of 55th and Woodlawn, where a lone gunman'
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
[..., ('at', 0.0044685714), ('the', 0.005466637), ('1700', 0.7173856), ('block', 0.81395197), ('of', 0.82227415), ('S.', 0.7940061), ('Halsted', 0.70529455), ('Ave.', 0.60538065), ...]
>>> geoextractor.extract_geostrings(article_text, prob_thresh=0.5)
[['1700', 'block', 'of', 'S.', 'Halsted', 'Ave.'], ['55th', 'and', 'Woodlawn,']]
```

## From the command line

The installation comes with a *very* rudimentary command line interface, which without any arguments defaults to reading from the stdin.

```bash
$ python -m tagnews.crimetype.cli
Go ahead and start typing. Hit ctrl-d when done.
<type here>
```

Or you can provide a list of articles to tag, a CSV of the probability of each tag is output to `<article name>.tagged`.

```bash
$ python -m tagnews.crimetype.cli sample-article-1.txt sample-article-2.txt
$ cat sample-article-1.txt.tagged
  CPD, 0.912382307
UNSPC, 0.051873838
 SEXA, 0.031065436
 BEAT, 0.023119570
 DRUG, 0.017140532
...
```

Note that the `-m` flag is required.

# Background

We want to compare the amount different types of crimes are reported in certain areas vs. the actual occurrence amount in those areas. Are some crimes under-represented in certain areas but over-represented in others? To accomplish this, we'll need to be able to extract a type-of-crime tag and geospatial data from news articles.

We meet every Tuesday at [Chi Hack Night](https://chihacknight.org/), and you can find out more about [this specific project here](https://github.com/chihacknight/breakout-groups/issues/61).

The [Chicago Justice Project](http://chicagojustice.org/) has been scraping RSS feeds of articles written by Chicago area news outlets for several years, allowing them to collect over 400,000 articles. At the same time, an amazing group of [volunteers](http://chicagojustice.org/volunteer-for-cjp/) have helped them tag these articles. The tags include crime categories like "Gun Violence", "Drugs", "Sexual Assault", but also organizations such as "Cook County State's Attorney's Office", "Illinois State Police", "Chicago Police Department", and other miscellaneous categories such as "LGBTQ", "Immigration". The volunteer UI was also recently updated to allow highlighting of geographic information.

# Contributing

You want to contribute? Great! Check out the [CONTRIBUTING.md](./CONTRIBUTING.md) file for more info.

# Areas of research

## Type-of-Crime Article Tagging

This part of this project aims to automate the category tagging using a specific branch of Machine Learning known as Natural Language Processing.

Possible models to use (some of which we have tried!) include

* [Bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model)
* [*n*-gram models](https://en.wikipedia.org/wiki/N-gram)
* [A combination of bag-of-words and *n*-gram models](http://www.inference.phy.cam.ac.uk/hmw26/papers/nescai2006.pdf)
* [Word Vectorization as a pre-processing step](https://www.tensorflow.org/tutorials/word2vec/)
* [Convolutional Neural Networks](https://arxiv.org/pdf/1504.01255v3.pdf)
* [Recurrent Neural Networks](http://www.cs.toronto.edu/~graves/preprint.pdf)

It might be useful to have an additional corpus of news articles that we can use for unsupervised feature learning without having to worry about over-fitting.

## Automated Geolocation

We also need to automatically find the geographic area of the crime the article is talking about. We have just recently updated the tagging interface to also allow highlighting geospatial information inside of articles and are collecting ground truth data. Once we have collected this data, we need to automate the process of detecting location information inside articles. An important note, we are relying on the power of current geocoders to take unstructured location information and output a latitude/longitude pair.

One possible path forward appeared to involve an approach developed by  [Everyblock](http://www.everyblock.com/). They got funding from the [Knight Foundation](http://www.knightfoundation.org/) to geolocate news articles and were required to open source their code. A brief investigation seems to show that their geolocating is actually just a giant [Regular Expression](https://github.com/kbrose/everyblock/blob/master/ebdata/ebdata/nlp/addresses.py). Investigation showed that it was not accurate enough on its own for our purposes.

Things to checkout:

* [Mapzen](https://mapzen.com/)
* [US Adress parser](https://github.com/datamade/usaddress)
* [libpostal](https://github.com/openvenues/libpostal)

## Things to consider

Some articles may discuss multiple crimes. Some crimes may occur in multiple areas, whereas others may not be associated with any geographic information (e.g. some kinds of fraud).

# See Also

* [Chicago Justice Project](http://chicagojustice.org/)
* [Database Repo](https://github.com/kyaroch/chicago-justice)
* [Chi Hack Night Group Description](https://github.com/chihacknight/breakout-groups/issues/61)
