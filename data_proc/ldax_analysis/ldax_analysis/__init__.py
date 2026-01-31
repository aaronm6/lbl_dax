from ._version import __version__
from .analysis_utils import *
from .proc_modules import *

del analysis_utils
del proc_modules

version = __version__
version_tuple = __version_tuple__ = tuple([int(item) for item in __version__.split('.')])
