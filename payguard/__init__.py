"""PayGuard: finds duplicate, split and suspicious payments in accounts-payable data."""
__version__ = "1.0.0"

from .data import Dataset, DataError, load_folder, prepare  # noqa: E402,F401
from .rules import Settings  # noqa: E402,F401
from .scoring import Result, analyse  # noqa: E402,F401
