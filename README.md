# Overview

Let's compare the amount different types of crimes are reported in certain areas vs. the actual occurrence amount in those areas. Are some crimes under-represented in certain areas but over-represented in others? To accomplish this, we'll need to be able to extract type-of-crime and geospatial data from news articles. Find out [more about this project here](https://github.com/chihacknight/breakout-groups/issues/61), and read more about [Chi Hack Night here](https://chihacknight.org/).

# The Code

Under the `src` folder you can find the source code.

The `load_data.py` file will load the data from the CSV files (stored not in GitHub). Specifically, look at the `load_data.load_data()` method, this returns a `k`-hot encoded tagging and the article data.
