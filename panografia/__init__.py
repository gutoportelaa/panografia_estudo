"""Panografia: costura de imagens transladadas."""

from .pipeline import StitchError, StitchParams, StitchResult, stitch

__all__ = ["StitchError", "StitchParams", "StitchResult", "stitch"]
__version__ = "0.2.0"
