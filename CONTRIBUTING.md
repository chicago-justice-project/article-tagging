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

To do geostring extraction (see below), you will also need the pre-trained word vectorizer GloVE which can be downloaded at http://nlp.stanford.edu/data/glove.6B.zip.

Finally, you will need to make sure you have the community area boundaries geojson file. This can be downloaded from [https://data.cityofchicago.org/Facilities-Geographic-Boundaries/CommAreas/igwz-8jzy/](https://data.cityofchicago.org/Facilities-Geographic-Boundaries/CommAreas/igwz-8jzy/) (use the export button). Put the downloaded file in the same data folder. Make sure the file is named "Boundaries - Community Areas (current).geojson".

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

### What is it?

The type-of-crime model builds a multi-class classifier that takes in text from a news article and for each type-of-crime tag outputs a probability that the tag applies to the news article. In other words, it tries to guess what kinds of crimes the news article discusses.

The model code can be found in `lib/tagnes/crimetype/models/binary_stemmed_logistic/save_model.py`. It's less than 100 lines, don't be afraid to read it!

The model relies on NLTK as a tokenizer and builds a binary bag-of-words vectorizer with 40000 features. (We restricted to 40000 features because performance did not decrease significantly and it made the model much smaller, useful when trying to publish to pypi as a package.) The vectorized versions of the articles are then used as input to a separate logistic regression for each crime tag.

### How to train it?

The `save_model.py` can be run as a script to save the trained model. To run it, `cd` into the `lib` directory and run

```
python -m tagnews.crimetype.models.binary_stemmed_logistic.save_model
```

The vectorizer is saved in the same directory as the code with the name `vectorizer-<year><month><day>-<hour><minute><second>.pkl`. The model is saved similarly, but with `model` instead of `vectorizer`.

This code trains on the whole labeled dataset. During development, the `lib/tagnews/crimetype/benchmark.py` file was used to perform cross validation.

### How to measure performance?

We never defined a single number that could be used to decide if one model was better than another, even though that's usually a critical step. We generated FPR/TPRs for all the crime categories and plotted those. The best way may be to fix an acceptable FPR rate at something like 5% or 10% and see what maximizes the mean TPR across a set of desired categories. In short, there's not a solid answer here and refining this would be super helpful in its own right.

### How might it be improved?

* Use a better vectorizer than bag-of-words, e.g. GloVe as used for the geostring model.
* We briefly tried a naive bayes classifier over a logistic regression and it didn't seem to improve performance, but naive bayes is usually used as the baseline for these kinds of tasks. Could it be made to work better?
* Add more examples of articles that have *no* tags. Right now we randomly sample 3000 such articles, but we could probably use more. This may help with an observed problem where some sports articles have a high chance of being about a crime according to the model (likely due to the high use of words like "shoot").

## The geostring extractor model

### What is it?

The geostring model builds a word-by-word probability that each word is part of a "geostring". A "geostring" is a list of words that define a location. They can be pretty accurate street addresses as in "the shooting happened at the *corner of 55th and Woodlawn*" or fuzzier locations such as a neighborhood name, a church name, etc. The per-word probability can be thresholded and we take all consecutive list of words above the threshold as the geostrings inside an article.

The model code can be found in `lib/tagnews/geoloc/models/lstm/save_model.py`. It's 150 lines of python code a good portion of which is trying to hit an external internet API. The keras library is used extensively.

The model relies on the pre-trained semantic word vectorizer GloVE to get a 50 dimensional feature vector for each word, and then a two layer bi-directional LSTM is used to generate the probabilities.

### How to train it?

The `save_model.py` file can be run as a script to save the trained model. To run it, `cd` into the `lib` directory and run

```
python -m tagnews.geoloc.models.lstm.save_model
```

The model is saved under `lib/tagnews/geoloc/models/lstm/saved/weights-*.hdf5`. The code will run for a set number of training epochs (one epoch is one pass through all of the training examples), saving the weights after each epoch.

### How to measure performance?

Download the validation data from https://geo-extract-tester.herokuapp.com/ (there is also training data available for downloading). Follow the instructions on that website to upload guesses and the ROC curve will be shown for your model's predictions. If you have a higher AUC than the current high score, congratulations! Please submit a Pull Request!

You can also upload your model's predictions via an API. There is code inside `lib/tagnews/geoloc/models/lstm/save_model.py` demonstrating this.

### How might it be improved?

* Including "naive" models that do simple look-ups against Chicago street names.
* Using a word vectorizer that handles out-of-vocabulary predictions better (perhaps `FastText`?).
* Just use a character-level CNN?
* Augment the training data by labeling more articles (see the "I want to contribute to Chicago Justice Project but I don’t want to work on this NLP stuff. What can I do?" section).

## The geocoding

### What is it?

Geocoding here refers to the process of sending a geostring (e.g. "55th and Woodlawn") to an external service to retrieve a best-guess latitude/longitude pair of where that geostring is referring to.

Right now, the geocoding is done using an instance of pelias hosted by CJP.

The code can be found in `lib/tagnews/geoloc/tag.py`, in the `get_lat_longs_from_geostrings` function.

### How might it be improved?

* Improve post-processing of geostrings (we do rudimentary things like append "Chicago, Illinois", but we could get more sophisticated).
* Improve the inputs to it by improving the geostring model.
* Improve the inputs by making a better post-processor of geostrings.
* Improve the confidence score.

### What if it breaks?

The last time the geocoding broke it was because they started checking for browser-like headers, so we updated our requests to have browser-like headers. Something like this may happen again and unfortunately there's no real playbook here.

The good news is that the geostrings will always be there, and if needed we can always re-process any geocoding that doesn't work.

## Testing

### The test suite

You can find the tests `lib/tagnews/tests/`. We use `pytest` as the test runner. The test coverage isn't phenomenal, but it's not terrible either. We always welcome Pull Requests making more and better tests!

### Running locally

You need the data to run the tests. If you have the data, great! You should be able to run the tests. If you don't have the data, you can copy the tiny subset of the data stored in `lib/tagnews/data/ci-data/` to `lib/tagnews/data/`. Make sure you have downloaded GloVE from http://nlp.stanford.edu/data/glove.6B.zip (and extracted it, etc.).

Beware that if you run the tests with the full data-set, it can take a _long_ time and a _lot_ of memory.

If you don't already have a type-of-crime or geostring model, you will need train one (see above).

Once that's completed, `cd` into the lib directory and run

```
python -m pytest --cov-report term-missing --cov=tagnews
```

### Continuous Integration Testing

We use [Travis CI](https://travis-ci.org/chicago-justice-project/article-tagging) for continuous integration testing. Any Pull Request will automatically have the test suite run, and any commit to the master branch will automatically have the test suite run.

This is configured via the `.travis.yml` file at the top-level of this project.

## Documentation

### How to write it?

Write it in this very file! Or the README.md file!

### How to publish it?

Documentation is not currently published. If you have interest in helping with this, submit a Pull Request!

## Publishing a new version to pypi

First, update the `__version__` variable in `lib/tagnews/__init__.py`, initially start out by bumping the version and making it a release candidate, e.g. `1.1.0rc1`.

Second, make sure the saved models either match the previously published version exactly (by downloading the current release, extracting it, and copying the model file to where it needs to be), or are _meant_ to be updated. Make sure only the saved model you want exists in your project, delete all others.

Then, use the following two commands to publish the new version:

```bash
python setup.py sdist
twine upload dist/tagnews-version.number.you.want.to.upload.tar.gz
```

Create a new anaconda environment to download the version for rudimentary testing. The Continuous Integration should take care of most rigorous testing, this is just to make sure everything is working. I usually run through the example at the top of the README.

Once you are happy, remove the `rc*` suffix and publish as the actual version. You should then create a [release](https://github.com/chicago-justice-project/article-tagging/releases) on GitHub, attempting to log all the changes and attach the tarball created by `python setup.py sdist`.

*Note: pypi has a limit on the size of projects that can be uploaded, and pypi was recently migrated to a new data warehouse. We originally had to request a size increase in [this issue](https://github.com/pypa/packaging-problems/issues/119).*

## I want to contribute to Chicago Justice Project but I don’t want to work on this NLP stuff. What can I do?

You can help out the [the team scraping articles/maintaining the volunteers' web interface](https://github.com/chicago-justice-project/chicago-justice). If that doesn't sound interesting either, we can always use more [volunteer taggers](http://chicagojustice.org/volunteer-for-cjp/). Or just show up Tuesday nights at ChiHackNight and ask what you can do!
