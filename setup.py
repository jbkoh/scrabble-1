#from distutils.core import setup
from setuptools import setup, find_packages
__author__ = 'Jason Koh'
__version__ = '0.0.1'
setup (
    name = 'scrabble',
    version = __version__,
    author = __author__,
    packages = ['scrabble'],
    description = 'Scrabble for building metadata normalization',
    include_package_data = True,
    install_requires = ['setuptools', 'requests', 'arrow']
)
