from typing import Dict, Type

from .base import OutputFormatter
from .csv import CsvFormatter
from .cetz import CeTZFormatter


class FormatterRegistry:
    """Registry pattern for output formatters."""

    _registry: Dict[str, Type[OutputFormatter]] = {
        "csv": CsvFormatter,
        "cetz": CeTZFormatter,
    }

    @classmethod
    def available_formats(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._registry.keys()))

    @classmethod
    def register(cls, name: str, formatter_cls: Type[OutputFormatter]) -> None:
        cls._registry[name.lower()] = formatter_cls

    @classmethod
    def get(cls, name: str) -> OutputFormatter:
        fmt_cls = cls._registry.get(name.lower())
        if fmt_cls is None:
            supported = ", ".join(cls.available_formats())
            raise ValueError(f"Unsupported format '{name}'. Available formats: {supported}")
        return fmt_cls()
