"""
seekrtools
Seekrtools is a python library layer that interfaces with SEEKR tools such as 
OpenMMVT and provides a number of useful utilities and extensions.
"""

# Add imports here
from .utilities import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
