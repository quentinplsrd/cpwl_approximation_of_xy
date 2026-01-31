from .utils import *
# from .build_pywraplp import *
from ._version import *
from .user import *
from .model import *
from .logging_config import *

__appname__ = "TempRegPy"
__version__ = "0.1.1"
__author__ = "Matija Pavičević"

(
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
    DB_READ_ERROR,
    DB_WRITE_ERROR,
    JSON_ERROR,
    ID_ERROR,
) = range(7)

ERRORS = {
    DIR_ERROR: "config directory error",
    FILE_ERROR: "config file error",
    DB_READ_ERROR: "database read error",
    DB_WRITE_ERROR: "database write error",
    ID_ERROR: "id error",
}