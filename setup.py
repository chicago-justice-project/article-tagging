#!/usr/bin/env python

from distutils.core import setup

from setuptools import setup, find_packages
from setuptools.command.install import install as _install


setup(name='tagnews',
      version='1.0.1',
      description=('automatically tag articles with justice-related categories'
                   ' and extract location information'),
      author='Kevin Rose',
      url='https://github.com/chicago-justice-project/article-tagging',
      package_dir={'': 'lib'},
      packages=['tagnews',
                'tagnews.utils',
                'tagnews.crimetype',
                'tagnews.crimetype.models.binary_stemmed_logistic',
                'tagnews.geoloc',
                'tagnews.geoloc.models.lstm',
                'tagnews.tests'],
      install_requires=['nltk', 'numpy>=1.13', 'scikit-learn==0.19.0', 'pandas', 'scipy',
                        'tensorflow>=1.4', 'keras'],
    #   tests_require=['pytest'],
      package_data={'tagnews': ['crimetype/models/binary_stemmed_logistic/*.pkl',
                                'geoloc/models/lstm/saved/*.hdf5',
                                'data/glove.6B.50d.txt']},
      python_requires=">=3.5", # for now
      zip_safe=False,
     )
