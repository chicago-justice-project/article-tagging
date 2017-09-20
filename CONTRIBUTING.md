# Setup

Fork this repo, clone your fork, and navigate to it. If you're going to be developing it doesn't necessarily make sense to install as a package, but you'll still need to install the dependencies. If you have not already installed the dependencies, then see the instructions in the [README](README.md).

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

# Getting Started

A good place to start is the [notebooks](./lib/notebooks). We recommend starting with the explorations notebook -- it should give you a sense of what the data looks like. After reading through that, the bag-of-words-count-stemmed-binary notebook should give you a sense of what the NLP model for tagging looks like. Reading through these should help you get up to speed, and running them is a pretty good test to make sure everything is installed correctly.

# What can I do?

It's important to keep in mind that it can take a significant amount of time to make sure everything is installed and working correctly, and to get a handle on everything that's going on. It's normal to be confused and have questions. Once you feel comfortable with things, then you can:

Check out the [open issues](https://github.com/chicago-justice-project/article-tagging/issues) and see if there's anything you'd like to tackle there.

If not, you can try and improve upon the existing model(s), but be warned, measuring performance in a multi-label task is non-trivial. See the `bag-of-words-count-stemmed-binary.ipynb` notebook for an attempt at doing so. Tweaking that notebook and seeing how performance changes might be a good place to start tinkering with the NLP code. You can also read the `tagnews.crimetype.benchmark.py` file to get an idea of how the cross validation is being performed.

Further yet, you can help improve this very documentation.

# FAQ

### Do I have to use a specific language to participate in article-tagging?

Thusfar, most of the work has been done in Python and R, but there's no reason that always has to be the case. If there is another language that would be perfect for this project or that you have expertise in, that works too. Talk with us and we can figure something out.

### Are there concepts that will be helpful for me to understand?

Definitely!  [This sklearn user guide](http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) details a number of the text analysis methodologies this project uses (sklearn is a Python library, but the user guide is great for understanding machine learning text analysis in general).  Also, see the section on 'Automated Article Tagging' in the [README](./README.md) for more detailed literature on some of the relevant concepts. Reading the code and looking up concepts you are unfamiliar with is a valid path forward as well!

### I want to contribute to Chicago Justice Project but I don’t want to work on this NLP stuff. What can I do?

You can help out the [the team scraping articles/maintaining the volunteers' web interface](https://github.com/chicago-justice-project/chicago-justice). If that doesn't sound interesting either, we can always use more [volunteer taggers](http://chicagojustice.org/volunteer-for-cjp/). Or just show up Tuesday nights and ask what you can do!

### How do I productize a model?

You [pickle](https://docs.python.org/3.6/library/pickle.html) it. But working with pickle is difficult. In order to sanely be able load things, I'm running python files that pickle the model using the `-m` flag, e.g. `python -m tagnews.crimetype.models.binary_stemmed_logistic.model` will run code that generates the pickles of the model. All modules should be imported in the same way they will exist when unpickling the model from `tagnews.crimetype.tag`.
