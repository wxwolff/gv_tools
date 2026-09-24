"""Configuration, adapter discovery, and dependency inspection."""

from ..environment import DEPENDENCY_GROUPS, dependency_report
from ..registry import available_adapters, register_adapter

__all__ = [
    "DEPENDENCY_GROUPS",
    "available_adapters",
    "dependency_report",
    "register_adapter",
]
