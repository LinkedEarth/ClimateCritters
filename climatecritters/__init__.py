from . import utils
from .model_critters import *          # all concrete models at cc.*
from .core import Forcing, CCModel, CCOutput   # top-level abstractions
from .core import forcing as forcing   # builder namespace at cc.forcing.*
# from .utils import *

# get the version
from importlib.metadata import version
__version__ = version('climatecritters')