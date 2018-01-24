# Setup

Fork this repo, clone your fork, and navigate to it. If you're going to be developing it doesn't necessarily make sense to install as a package, but you'll still need to install the dependencies:

## Dependencies

***This code requires python 3.5 or greater.***

Additionally, to use this code, you will need at least the python packages

* [nltk](http://www.nltk.org/),
* [numpy](http://www.numpy.org/) at version 1.13 or higher,
* [scikit-learn](http://scikit-learn.org/),
* [tensorflow](https://www.tensorflow.org/) at version 1.4 or greater,
* [keras](https://keras.io/), and
* [pandas](http://pandas.pydata.org/).

If you need detailed instructions, see the "How do I get the dependencies?" section in the FAQ below.

(See the `install_requires` line in setup.py for the definitive list.)


## The Data

Due mainly to the file size, the data is not included in the GitHub repo. Instead, it is available on a USB drive at Chi Hack Night, so you'll need to get it there. Extract the data from the archive on the USB drive. Copy the contents into the folder `lib/tagnews/data/`. After this is done, your directory should look something like this:

```bash
(cjp-at) .../article-tagging/lib/tagnews/data$ ls -l
total 2117928
-rw-r--r-- 1 kevin.rose 1049089       6071 Sep 19 23:45 column_names.txt
-rw-r--r-- 1 kevin.rose 1049089 2156442023 Sep 18 21:02 newsarticles_article.csv
-rw-r--r-- 1 kevin.rose 1049089       2642 Sep 18 21:02 newsarticles_category.csv
-rw-r--r-- 1 kevin.rose 1049089   10569986 Sep 18 21:02 newsarticles_usercoding.csv
-rw-r--r-- 1 kevin.rose 1049089    1726739 Sep 18 21:02 newsarticles_usercoding_categories.csv
```

Once extracted to the correct place, you can load the data as follows:

```python
>>> import tagnews
>>> df = tagnews.load_data(nrows=None) # change to int to load subset
```

# Getting Started

A good place to start is the [notebooks](./lib/notebooks). We recommend starting with the explorations notebook -- it should give you a sense of what the data looks like. After reading through that, the bag-of-words-count-stemmed-binary notebook should give you a sense of what the NLP model for tagging looks like. Reading through these should help you get up to speed, and running them is a pretty good test to make sure everything is installed correctly.

# What can I do?

It's important to keep in mind that it can take a significant amount of time to make sure everything is installed and working correctly, and to get a handle on everything that's going on. It's normal to be confused and have questions. Once you feel comfortable with things, then you can:

Check out the [open issues](https://github.com/chicago-justice-project/article-tagging/issues) and see if there's anything you'd like to tackle there.

If not, you can try and improve upon the existing model(s), but be warned, measuring performance in a multi-label task is non-trivial. See the `bag-of-words-count-stemmed-binary.ipynb` notebook for an attempt at doing so. Tweaking that notebook and seeing how performance changes might be a good place to start tinkering with the NLP code. You can also read the `tagnews.crimetype.benchmark.py` file to get an idea of how the cross validation is being performed.

Further yet, you can help improve this very documentation.

# FAQ

### How do I get the dependencies?

If you are having trouble installing the requirements, then we recommend using [Anaconda](https://www.continuum.io/downloads) to manage python environments. If you are unfamiliar with Anaconda, you should read about it at the linked site above.

Once it is installed, you can create a new environment. If you are using bash (mac or linux):

```bash
$ # create a new anaconda environment with required packages
$ conda create -n cjp-at "python>=3.5" nltk "numpy>=1.13" scikit-learn pandas pytest
$ source activate cjp-at
(cjp-at) $ pip install "tensorflow>=1.4"
(cjp-at) $ pip install keras
(cjp-at) > ...
```

If you are using cmd (windows):

```cmd
> conda create -n cjp-at "python>=3.5" nltk "numpy>=1.13" scikit-learn pandas pytest
> activate cjp-at
(cjp-at) $ pip install "tensorflow>=1.4"
(cjp-at) $ pip install keras
(cjp-at) > ...
```

If you have an NVIDIA GPU on your machine then you may wish to use `pip install "tensorflow-gpu>=1.4"` instead.

### How do I fix errors from NLTK about missing data?
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

then you need to download some nltk data. The easiest way to download nltk data is to just run

```python
>>> import nltk
>>> nltk.download()
```

and use the GUI. If you wish to do this programatically, then you can run `nltk.download('corpus_name')`. Right now there are only two dependencies:

```python
>>> nltk.download('punkt')
>>> nltk.download('wordnet')
```

### Do I have to use a specific language to participate in article-tagging?

Thusfar, most of the work has been done in Python and R, but there's no reason that always has to be the case. If there is another language that would be perfect for this project or that you have expertise in, that works too. Talk with us and we can figure something out.

### Are there concepts that will be helpful for me to understand?

Definitely!  [This sklearn user guide](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) details a number of the text analysis methodologies this project uses (sklearn is a Python library, but the user guide is great for understanding machine learning text analysis in general).  Also, see the section on 'Automated Article Tagging' in the [README](./README.md) for more detailed literature on some of the relevant concepts. Reading the code and looking up concepts you are unfamiliar with is a valid path forward as well!

### I want to contribute to Chicago Justice Project but I don’t want to work on this NLP stuff. What can I do?

You can help out the [the team scraping articles/maintaining the volunteers' web interface](https://github.com/chicago-justice-project/chicago-justice). If that doesn't sound interesting either, we can always use more [volunteer taggers](http://chicagojustice.org/volunteer-for-cjp/). Or just show up Tuesday nights and ask what you can do!

### How do I productize a model?

You [pickle](https://docs.python.org/3.6/library/pickle.html) it. But working with pickle is difficult. In order to sanely be able load things, I'm running python files that pickle the model using the `-m` flag, e.g. `python -m tagnews.crimetype.models.binary_stemmed_logistic.save_model` will run code that generates the pickles of the model. (Note that you need to be in the `lib` folder to do that.) All modules should be imported in the same way they will exist when unpickling the model from `tagnews.crimetype.tag`.

### How is this published to pypi?

```bash
python setup.py sdist
twine upload dist/tagnews-version.number.you.want.to.upload.tar.gz
```
