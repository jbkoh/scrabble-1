#from distutils.core import setup
from setuptools import setup, find_packages
__author__ = 'Jason Koh'
__version__ = '0.0.1'
setup (
    name = 'scrabble',
    version = __version__,
    author = __author__,
    #packages = ['scrabble'],
    packages = find_packages(),
    description = 'Scrabble for building metadata normalization',
    include_package_data = True,
    data_files=[
        ('metadata', ['metadata/relationship_prior.json',
                      'metadata/bacnettype_mapping.csv',
                      'metadata/unit_mapping.csv',
                      ]),
        ('brick', ['brick/tags.json',
                   'brick/equip_tagsets.json',
                   'brick/location_tagsets.json',
                   'brick/point_tagsets.json',
                   'brick/location_subclass_dict.json',
                   'brick/point_subclass_dict.json',
                   'brick/equip_subclass_dict.json',
                   'brick/tagset_tree.json',
                   ])
    ],
    install_requires = ['setuptools', 'requests', 'arrow']
)
