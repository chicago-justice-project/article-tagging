#!/usr/bin/env python

from setuptools import setup

import os

with open('README.md', "r") as f:
    long_description = f.read()


init_file = os.path.join(os.path.split(__file__)[0], 'lib/tagnews/__init__.py')
with open(init_file) as f:
    try:
        s = f.read()
        version_index = s.index('__version__')
        version = s[version_index:].split('\n')[0].split("'")[1].strip()
        # make sure it is in correct format by trying to parse it
        assert len(version.split('.')) == 3
    except Exception as e:
        raise RuntimeError(
            'Problem parsing lib/tagnews/__init__.py to get version.'
            ' Make sure somewhere in that file there is a line that'
            ' looks approximately like "__version__ = \'x.y.z\'",'
            ' including using single quotes, not double quotes.'
        )

setup(name='tagnews',
      version=version,
      description=('automatically tag news articles with justice-related'
                   ' categories and extract location information'),
      author='Kevin Rose, Josh Herzberg, Matt Sweeney',
      url='https://github.com/chicago-justice-project/article-tagging',
      package_dir={'': 'lib'},
      packages=['tagnews',
                'tagnews.utils',
                'tagnews.crimetype',
                'tagnews.crimetype.models.binary_stemmed_logistic',
                'tagnews.geoloc',
                'tagnews.geoloc.models.lstm',
                'tagnews.senteval',
                'tagnews.tests'],
      install_requires=['nltk==3.2.5', 'numpy==1.14', 'scikit-learn==0.19.0',
                        'pandas==0.22.0', 'scipy==1.0.0', 'tensorflow==1.5',
                        'h5py==2.7.1', 'keras==2.1.4', 'shapely==1.6.4.post2',
                        'requests==2.18.4', 'google-cloud-language==1.3.0'],
      package_data={'tagnews': ['crimetype/models/binary_stemmed_logistic/*.pkl',
                                'geoloc/models/lstm/saved/*.hdf5',
                                'data/glove.6B.50d.txt',
                                'data/Boundaries - Community Areas (current).geojson']},
      python_requires=">=3.5",
      zip_safe=False,
      long_description=long_description,
      long_description_content_type='text/markdown',
     )
