"""Common type aliases used across the LightGBM regression toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

__all__ = ["PathLike"]
