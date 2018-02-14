# Geotagging Quick Start Instructions

Want to help out our group? Start here!

The Chicago Justice Project's Quantifying Justice News Project aims to identify  discrepancies in media reporting of crime. To do this with a computer instead of by hand, we

1. Scrape crime articles from the web.
2. Identify what "type of crime" the article describes.
3. Identify where the crime occured. This part has two steps.

  1. Identify which words describe the location of the crime.*
  2. Pass this string through [a python geocoder package](https://pypi.python.org/pypi/geocoder).*

\*This is where you can help!

# Installing
1. Start by cloning/downloading the repository. [Follow Kevin's instruction's](CONTRIBUTING.md).
2. Download the [GloVe](https://nlp.stanford.edu/projects/glove/) pretrained word vectors from Stanford. Put this in lib/tagnews/data.
3. Also download the [ner.csv](https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus/downloads/ner.csv) dataset from kaggle. Put this in the same place as 2.
4. Open up the jupyter notebook called [extract-geostring-example.ipynb](lib/notebooks).
5. Have fun!

If you wish to train the existing model, first of all make sure the data and package dependencies are all set, and then run the following command from the `lib` directory:

```
python -m tagnews.geoloc.models.lstm.save_model
```

This will train and save a model. A saved model can be loaded with

```python
>>> import tagnews
>>> geoextractor = tagnews.GeoCoder()
>>> article_text = 'The murder occurred at the 1700 block of S. Halsted. It happened just after midnight. Another murder occurred at the intersection of 55th and Woodlawn, where a lone gunman...'
>>> geoextractor.extract_geostrings(article_text)
[['1700', 'block', 'of', 'S.', 'Halsted.'], ['intersection', 'of', '55th', 'and', 'Woodlawn,']]
>>> prob_out = geoextractor.extract_geostring_probs(article_text)
>>> import matplotlib.pyplot as plt
>>> plt.plot(prob_out[1])
>>> for x, text in enumerate(prob_out[0]):
...     plt.text(x, -.2, text, horizontalalignment='center')
>>> plt.ylim([-.4, 1])
>>> plt.show()
```

# a little more detail
1. Don't have python? Check out [anaconda](https://conda.io/docs/user-guide/install/index.html).
2. Don't have git? Check out [git](https://git-scm.com/downloads).
3. Never used jupyter notebooks? [Try them out](http://jupyter.readthedocs.io/en/latest/install.html)!
4. Never used pandas? [Ten minutes to pandas](https://pandas.pydata.org/pandas-docs/stable/10min.html) is a pretty helpful tutorial.

**What's going on here?**

The process of extracting the address strings is called [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition), which is a subset of [Natural Language Processing](https://en.wikipedia.org/wiki/natural_language_processing) . For example,

[Jim]<sub>Person</sub> bought 300 shares of [Acme Corp.]<sub>Organization</sub> in [2006]<sub>Time</sub>.

We want to recognize the location entities in the crime articles.

How do we train a model with words as inputs? The GloVe vectors are word embeddings. They are vectors that represent words and the relationships between them. For example, in the Glove link, you can see the first picture shows the `man`, `woman`, `king`, and `queen` vectors (in 2D). What's cool and useful is the capability of the vectors to contain relationships between words that reflect their semantic definitions. In this example, `king - man + woman = queen`.

Using these vectors, we can train a model to pick out which vectors represent location words. [scikit-learn](http://scikit-learn.org/) makes some models fairly easy to try out.

**How can I improve on what has been done already?**

There are a few obvious ways forward, feel free to pick and choose what you think will make the greatest improvement.
1. Improve the dataset. Above I instructed you to download the ner.csv dataset from kaggle. This is definitely not the only NER dataset! There may be a much better one! See the [github issue](https://github.com/chicago-justice-project/article-tagging/issues/62) to see what we've discussed.
2. Improve the model. The example notebook uses a Random Forest. There are definitely better models. Some take single word inputs. Some account for sequences of words. Use the Heroku app to beat the best score!
