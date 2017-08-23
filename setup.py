#!/usr/bin/env python

from distutils.core import setup

from setuptools import setup, find_packages
from setuptools.command.install import install as _install

required_nltk_packages = ['punkt', 'wordnet']

class Install(_install):
    def run(self):
        _install.do_egg_install(self)
        import nltk
        for nltk_package in required_nltk_packages:
            nltk.download(nltk_package)

setup(name='newstag',
      version='0.0.5',
      description='automatically tag articles with justice-related categories',
      author='Kevin Rose',
      url='https://github.com/chicago-justice-project/article-tagging',
      package_dir={'': 'lib'},
      packages=['newstag', 'newstag.utils', 'newstag.crimetype'],
      install_requires=['nltk', 'numpy', 'scikit-learn', 'pandas'],
      tests_require=['pytest'],
      package_data={'newstag': ['crimetype/models/binary_stemmed_logistic/*.pkl']},
      cmdclass={'install': Install},
      setup_requires=['nltk'],
      zip_safe=False, # force source installation
     )
