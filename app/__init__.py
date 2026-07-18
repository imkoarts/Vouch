"""Vouch local editorial application package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("vouch")
except PackageNotFoundError:
    __version__ = "0.19.7"

__runtime_revision__ = "email-voice-generation-reliability-20260718.13"

__all__ = ["__runtime_revision__", "__version__"]
