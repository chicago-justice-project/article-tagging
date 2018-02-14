# Installation

This document contains instructions for installing this library as a system package which can then be used to deliver NLP-based tagging results. If you are just looking to hack on the NLP/library, follow the instructions in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## Dependencies

***This code requires python 3.5 or greater.***

Additionally, to use this code, you will need at least the python packages [nltk](http://www.nltk.org/), [numpy](http://www.numpy.org/) at version 1.13 or higher, [scikit-learn](http://scikit-learn.org/), and [pandas](http://pandas.pydata.org/). If you need detailed instructions, see below.

## Pip Install

Once requirements are installed, simply pip install the package:

```
pip install tagnews
```

To give it a test run, try running the following:

```python
>>> import tagnews
>>> crimetags = tagnews.CrimeTags()
>>> article_text = 'A short article. About drugs and police.'
>>> crimetags.tagtext_proba(article_text)
DRUG     0.747944
CPD      0.617198
VIOL     0.183003
UNSPC    0.145019
ILSP     0.114254
POLM     0.059985
...
>>> crimetags.tagtext(article_text, prob_thresh=0.5)
['DRUG', 'CPD']
```

If you get an error that looks something like

```
Traceback (most recent call last):
  <snip>
LookupError:
**********************************************************************
  Resource 'tokenizers/punkt/PY3/english.pickle' not found.
  Please use the NLTK Downloader to obtain the resource:  >>>
  nltk.download()
  Searched in:
    - '/home/kevin/nltk_data'
    - '/usr/share/nltk_data'
    - '/usr/local/share/nltk_data'
    - '/usr/lib/nltk_data'
    - '/usr/local/lib/nltk_data'
    - ''
**********************************************************************
```

then you need to download some nltk data. See the NLTK section below for more information.

## Further Setup

### Requirements ctd.

If you are having trouble installing the requirements, then we recommend using [Anaconda](https://www.continuum.io/downloads) to manage python environments. If you are unfamiliar with Anaconda, you should read about it at the linked site above.

Once it is installed, you can create a new environment. If you are using bash (mac or linux):

```bash
$ # create a new anaconda environment with required packages
$ conda create -n cjp-at "python>=3.5" nltk "numpy>=1.13" scikit-learn pandas pytest
$ source activate cjp-at
(cjp-at) $ ...
```

If you are using cmd (windows):

```cmd
> conda create -n cjp-at "python>=3.5" nltk "numpy>=1.13" scikit-learn pandas pytest
> activate cjp-at
(cjp-at) > ...
```

### NLTK data

The easiest way to download nltk data is to just run

```python
>>> import nltk
>>> nltk.download()
```

and use the GUI. If you wish to do this programatically, then you can run `nltk.download('corpus_name')`. Right now there are only two dependencies:

```python
>>> nltk.download('punkt')
>>> nltk.download('wordnet')
```
