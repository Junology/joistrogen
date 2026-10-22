from typing import Dict, Type

from .base import OutputFormatter
from .csv import CsvFormatter


class FormatterRegistry:
    """Registry pattern for output formatters."""

    _registry: Dict[str, Type[OutputFormatter]] = {
        "csv": CsvFormatter,
    }

    @classmethod
    def register(cls, name: str, formatter_cls: Type[OutputFormatter]) -> None:
        cls._registry[name.lower()] = formatter_cls

    @classmethod
    def get(cls, name: str) -> OutputFormatter:
        fmt_cls = cls._registry.get(name.lower())
        if fmt_cls is None:
            supported = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unsupported format '{name}'. Available formats: {supported}")
        return fmt_cls()
