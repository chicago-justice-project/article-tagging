# Setup

You need two things to work on this project; the code, and the data.

## The Code

For development, we strongly recommend you use something like virtual environments in python or a conda environment.

To get the code,

1. On GitHub, fork this repo
2. `git clone` your fork
3. `cd` to it

Then you can run

```
pip install -e .
```

This will install the package in editable mode, and will download all the dependencies as a side effect. Changes you make to any of the source code files will be automatically picked up the next time you import `tagnews`.

You will likely need to install some [NLTK](http://www.nltk.org/) data as well:

```python
>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('wordnet')
>>> quit()
```

## The Data

The data used to be retrievable via an SFTP server, but this has since been shut down. You can follow [this issue](https://github.com/chicago-justice-project/chicago-justice/issues/74) to see progress on restarting the nightly database exports.

However you get the data, copy the contents into the folder `lib/tagnews/data/`. After this is done, your directory should look something like this:

```bash
.../article-tagging/lib/tagnews/data$ ls -l
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
>>> df = tagnews.load_data(nrows=10) # change to some None for all rows
```

# Directory structure

This project is structured as follows:

```
├───lib
│   ├───notebooks ............................ Jupyter/IPython notebooks
│   └───tagnews .............................. Python package/source code
│       ├───crimetype ........................ Code related to time-of-crime tagging
│       │   └───models ....................... Filler directory
│       │       └───binary_stemmed_logistic .. Code to train/save crimetype NLP model
│       ├───data ............................. Put the data in here!
│       │   └───ci-data ...................... A tiny subset of data used for testing
│       ├───geoloc ........................... Code related to geocoding
│       │   └───models ....................... Filler directory
│       │       └───lstm ..................... Code *and data* to train/save geostring extractor
│       │           └───saved ................ Where the geostring model is saved.
│       ├───tests ............................ Code used to test this project
│       └───utils ............................ Helper functions, mostly around data loading
└───r_models ................................. R code, unused for a while, use with caution
```

Depending on how you want to contribute will dictate which parts you need to know about.

# What can I do?

There are a couple things you could do, each item listed here is expounded on further below.

* Improve the type-of-crime model (article text -> type-of-crime tags)
* Improve the geostring extractor model (article text -> list of location strings)
* Improve the geocoding (list of location strings -> list of lat/longs)
* Write more tests
* Write documentation
* Ways to help without coding

## The type-of-crime model

## The geostring extractor model

## The geocoding

## Testing

## Documentation

# FAQ

### Are there concepts that will be helpful for me to understand?

Definitely!  [This sklearn user guide](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) details a number of the text analysis methodologies this project uses (sklearn is a Python library, but the user guide is great for understanding machine learning text analysis in general).  Also, see the section on 'Automated Article Tagging' in the [README](./README.md) for more detailed literature on some of the relevant concepts. Reading the code and looking up concepts you are unfamiliar with is a valid path forward as well!

### I want to contribute to Chicago Justice Project but I don’t want to work on this NLP stuff. What can I do?

You can help out the [the team scraping articles/maintaining the volunteers' web interface](https://github.com/chicago-justice-project/chicago-justice). If that doesn't sound interesting either, we can always use more [volunteer taggers](http://chicagojustice.org/volunteer-for-cjp/). Or just show up Tuesday nights and ask what you can do!

### How do I productize a model?

You [pickle](https://docs.python.org/3.6/library/pickle.html) it. But working with pickle is difficult. In order to sanely be able load things, I'm running python files that pickle the model using the `-m` flag, e.g. `python -m tagnews.crimetype.models.binary_stemmed_logistic.save_model` will run code that generates the pickles of the model. (Note that you need to be in the `lib` folder to do that.) All modules should be imported in the same way they will exist when unpickling the model from `tagnews.crimetype.tag`.

### How is this published to pypi?

First, update the `__version__` variable in `lib/tagnews/__init__.py`, initially start out by bumping the version and making it a release candidate, e.g. `1.1.0rc1`. Then, use the following two commands to publish the new version:

```bash
python setup.py sdist
twine upload dist/tagnews-version.number.you.want.to.upload.tar.gz
```

Create a new anaconda environment to download the version for rudimentary testing. The Continuous Integration should take care of most rigorous testing, this is just to make sure everything is working. I usually run through the example at the top of the README.

Once you are happy, remove the `rc*` suffix and publish as the actual version. You should then create a [release](https://github.com/chicago-justice-project/article-tagging/releases) on GitHub, attempting to log all the changes and attach the tarball created by `python setup.py sdist`.
