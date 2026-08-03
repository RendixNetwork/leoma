"""Leoma package metadata."""

from importlib.metadata import PackageNotFoundError, version


try:
    # pyproject.toml / installed distribution metadata is the single source of
    # truth. A duplicated literal here previously made the 0.3.9 image advertise
    # itself as 0.3.2 even though its wheel metadata was correct.
    __version__ = version("leoma")
except PackageNotFoundError:  # source tree imported before installation
    __version__ = "0+unknown"

__all__ = ["__version__"]
