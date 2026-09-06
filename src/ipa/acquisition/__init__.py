"""External acquisition and safety adapters (transitional facade)."""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "WebScraper": ("web_scraper", "WebScraper"),
    "ScrapeSite": ("web_scraper", "ScrapeSite"),
    "OCRAdapter": ("ocr_adapter", "OCRAdapter"),
    "AutoFetchStrategy": ("fetch_strategy", "AutoFetchStrategy"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(f"ipa.acquisition.{module_name}"), attr)


__all__ = sorted(_EXPORTS)

