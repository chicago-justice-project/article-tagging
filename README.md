# Overview

Let's compare the amount different types of crimes are reported in certain areas vs. the actual occurrence amount in those areas. Are some crimes under-represented in certain areas but over-represented in others? To accomplish this, we'll need to be able to extract type-of-crime tag and geospatial data from news articles. We meet every Tuesday at [Chi Hack Night](https://chihacknight.org/), and you can find out more about [this specific project here](https://github.com/chihacknight/breakout-groups/issues/61).

# Details

The [Chicago Justice Project](http://chicagojustice.org/) has been scraping RSS feeds of articles written by Chicago area news outlets for several years, allowing them to collect almost 300,000 articles. At the same time, an amazing group of [volunteers](http://chicagojustice.org/volunteer-for-cjp/) have helped them tag these articles. The tags include crime categories like "Gun Violence", "Drugs", "Sexual Assault", but also organizations such as "Cook County State's Attorney's Office", "Illinois State Police", "Chicago Police Department", and other miscellaneous categories such as "LGBTQ", "Immigration".

## Automated Article Tagging

This part of this project aims to automate the category tagging using a specific branch of Machine Learning known as Natural Language Processing.

Possible models to try include

* [Bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model)
* [*n*-gram models](https://en.wikipedia.org/wiki/N-gram)
* [A combination of bag-of-words and *n*-gram models](http://www.inference.phy.cam.ac.uk/hmw26/papers/nescai2006.pdf)
* [Word Vectorization as a pre-processing step](https://www.tensorflow.org/tutorials/word2vec/)
* [Convolutional Neural Networks](https://arxiv.org/pdf/1504.01255v3.pdf)
* [Recurrent Neural Networks](http://www.cs.toronto.edu/~graves/preprint.pdf)

It might be useful to have an additional corpus of news articles that we can use for unsupervised feature learning without having to worry about over-fitting.

## Automated Geolocation

We also need to automatically find the geographic area of the crime the article is talking about. A group called [Everyblock](http://www.everyblock.com/) got funding from the [Knight Foundation](http://www.knightfoundation.org/) to geolocate news articles. They were required to open source their code. A brief investigation seems to show that their geolocating is actually just a giant [Regular Expression](https://github.com/kbrose/everyblock/blob/master/ebdata/ebdata/nlp/addresses.py). Whether or not this will be good enough remains to be seen.

Unfortunately, we do not currently have labeled training data for this task, although it's possible there might be some open data source somewhere.

## Things to consider

Some articles may discuss multiple crimes. Some crimes may occur in multiple areas, whereas others may not be associated with any geographic information (e.g. some kinds of fraud).

# The Code

Under the `src` folder you can find the source code.

The `load_data.py` file will load the data from the CSV files (stored not in GitHub). Specifically, look at the `load_data.load_data()` method, this returns a `k`-hot encoded tagging and article data.

# See Also

* [Chicago Justice Project](http://chicagojustice.org/)
* [Database Repo](https://github.com/kyaroch/chicago-justice)
* [Chi Hack Night Group Description](https://github.com/chihacknight/breakout-groups/issues/61)
