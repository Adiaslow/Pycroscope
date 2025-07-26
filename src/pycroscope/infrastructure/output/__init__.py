"""
Output infrastructure for Pycroscope.

Handles file output, serialization, and result persistence
following clean architecture principles.
"""

from .output_manager import OutputManager

__all__ = ["OutputManager"]
