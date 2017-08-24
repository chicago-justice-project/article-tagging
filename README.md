# Installation and Usage

## Requirements

To install this library, you will need at least the python packages [nltk](http://www.nltk.org/), [numpy](http://www.numpy.org/), [scikit-learn](http://scikit-learn.org/), and [pandas](http://pandas.pydata.org/). We recommend using [Anaconda](https://www.continuum.io/downloads) to manage python environments:

```bash
$ # create a new anaconda environment with required packages
$ conda create -n article-tagging nltk numpy scikit-learn pandas pytest
$ source activate article-tagging
(article-tagging) $ ...
```

## Installation

Download the code from git, `cd` into the directory, and run the setup.py file. If you created an Anaconda environment, then make sure that environment is active before running the setup file.

```bash
$ git clone git@github.com:chicago-justice-project/article-tagging.git
$ cd article-tagging
$ python setup.py install
```

### nltk

As long as the nltk package is already installed, running the setup.py file should automatically download the required nltk corpora. If that does not work for some reason, then you will need to download the corpora manually. See the list `required_nltk_packages` in setup.py. Each corpus can be downloaded by running `nltk.download(corpus_name)`

## Testing

You will additionally need `pytest` installed to run the tests.

TODO

## Usage

### From python

The main class is `tagnews.crimetype.tag.Tagger`:

```python
>>> import tagnews
>>> tagger = tagnews.crimetype.tag.Tagger()
>>> article_text = 'A short article. About drugs and police.'
>>> tagger.relevant(article_text, prob_thresh=0.1)
True
>>> tagger.tagtext(article_text, prob_thresh=0.5)
['DRUG', 'CPD']
>>> tagger.tagtext_proba(article_text)
DRUG     0.747944
CPD      0.617198
VIOL     0.183003
UNSPC    0.145019
ILSP     0.114254
POLM     0.059985
...
```

### From the command line

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

The [Chicago Justice Project](http://chicagojustice.org/) has been scraping RSS feeds of articles written by Chicago area news outlets for several years, allowing them to collect almost 300,000 articles. At the same time, an amazing group of [volunteers](http://chicagojustice.org/volunteer-for-cjp/) have helped them tag these articles. The tags include crime categories like "Gun Violence", "Drugs", "Sexual Assault", but also organizations such as "Cook County State's Attorney's Office", "Illinois State Police", "Chicago Police Department", and other miscellaneous categories such as "LGBTQ", "Immigration". The volunteer UI was also recently updated to allow highlighting of geographic information.

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
