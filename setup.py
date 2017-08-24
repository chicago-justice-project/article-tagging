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

setup(name='tagnews',
      version='0.1.1',
      description='automatically tag articles with justice-related categories',
      author='Kevin Rose',
      url='https://github.com/chicago-justice-project/article-tagging',
      package_dir={'': 'lib'},
      packages=['tagnews',
                'tagnews.utils',
                'tagnews.crimetype',
                'tagnews.tests'],
      install_requires=['nltk', 'numpy', 'scikit-learn', 'pandas'],
      tests_require=['pytest'],
      package_data={'tagnews': ['crimetype/models/binary_stemmed_logistic/*.pkl',
                                'data/*.csv']},
      cmdclass={'install': Install},
      setup_requires=['nltk'],
      zip_safe=False, # force source installation
     )
