#!/usr/bin/env python

from distutils.core import setup

from setuptools import setup, find_packages
from setuptools.command.install import install as _install


setup(name='tagnews',
      version='0.6.0',
      description='automatically tag articles with justice-related categories',
      author='Kevin Rose',
      url='https://github.com/chicago-justice-project/article-tagging',
      package_dir={'': 'lib'},
      packages=['tagnews',
                'tagnews.utils',
                'tagnews.crimetype',
                'tagnews.crimetype.models.binary_stemmed_logistic',
                'tagnews.tests'],
      install_requires=['nltk', 'numpy>=1.13', 'scikit-learn==0.19.0', 'pandas', 'scipy'],
      tests_require=['pytest'],
      package_data={'tagnews': ['crimetype/models/binary_stemmed_logistic/*.pkl']},
      python_requires=">=3.5", # for now
      zip_safe=False, # force source installation
     )
