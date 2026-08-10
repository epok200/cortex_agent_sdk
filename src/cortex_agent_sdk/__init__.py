from importlib.metadata import version as _package_version

__version__ = _package_version("cortex-agent-sdk")

__all__ = ["__version__"]
