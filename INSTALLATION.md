# Installation

This document contains instructions for installing this library as a system package which can then be used to deliver NLP-based tagging results. If you are just looking to hack on the NLP/library, follow the instructions in the [README.md](README.md).

## Requirements

To use this code, you will need at least the python packages [nltk](http://www.nltk.org/), [numpy](http://www.numpy.org/), [scikit-learn](http://scikit-learn.org/), and [pandas](http://pandas.pydata.org/). We recommend using [Anaconda](https://www.continuum.io/downloads) to manage python environments, but this is by no means reuqired.

```bash
$ # create a new anaconda environment with required packages
$ conda create -n article-tagging nltk numpy scikit-learn pandas pytest
$ source activate article-tagging
(article-tagging) $ ...
```

## Setup

Download the code from git, `cd` into the directory, and run the setup.py file. If you created an Anaconda environment, then make sure that environment is active before running the setup file. Please make sure the nltk package is installed before running the setup.py file. See below for why.

```bash
$ git clone git@github.com:chicago-justice-project/article-tagging.git
$ cd article-tagging
$ python setup.py install
```

(If you do not want to clone, you could also `wget` the zip file GitHub provides.)

### nltk

As long as the nltk package is already installed, running the setup.py file should automatically download the required nltk corpora. If that does not work for some reason, or you do not want to run the setup.py file, then you will need to download the corpora manually. See the list `required_nltk_packages` in setup.py. Each corpus can be downloaded by running `nltk.download(corpus_name)`.

## The NLP Model

Pre-trained models are not saved in Git since they are large binary files. Saved models can be downloaded from INSERT DOWNLOAD LOCATION. Alternatively, models can be generated locally by running

```
python -m tagnews.crimetype.models.binary_stemmed_logistic.model
```

which will save two files, `model.pkl` and `vectorizer.pkl` to your current directory. Generating a model locally will require having the data (see below).

Wherever these two files end up being located (either by downloading or creating locally), you can reference this folder when creating your `Tagger` instance (see simple usage below).

## Data

The data is not stored in the Git repo since it would take up a considerable amount of space. Instead, the data is dumped daily on the server and can be accessed using a SFTP client. The data is only necessary if you wish to create your own model.

# Simple Usage

```python
>>> import tagnews
>>> tagger = tagnews.crimetype.tag.Tagger(model_directory='path/to/folder/containing/pickles/')
>>> article_text = 'A short article. About drugs and police.'
>>> tagger.tagtext_proba(article_text)
DRUG     0.747944
CPD      0.617198
VIOL     0.183003
UNSPC    0.145019
ILSP     0.114254
POLM     0.059985
...
```

# Testing

You will additionally need the pytest library installed to run the tests.

To test an installation, you can run

```python
import tagnews
tagnews.test()
```

to run the tests.
