"""Document conversion: Docling ``DocumentConverter().convert(path)`` on pdf/docx/xlsx/pptx.

Docling's defaults apply (OCR on image regions included). The answer carries the whole
``DoclingDocument`` as ``export_to_dict()``, so the client maps it exactly like a local conversion.
``docling`` is imported when the converter is first needed.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from vlm_server.gateway.deps import install_hint
from vlm_server.gateway.errors import ConversionFailed, EngineUnavailable

#: What the endpoint converts (MentorMind's Docling extractor takes the same four). Docling itself
#: would also read e.g. .txt as Markdown; the contract keeps to these.
SUFFIXES = (".pdf", ".docx", ".xlsx", ".pptx")


class ConvertedDocument(BaseModel):
    filename: str
    num_pages: int
    #: ``DoclingDocument.export_to_dict()``.
    document: dict[str, Any]


class DocumentEngine(Protocol):
    def convert(self, path: Path) -> tuple[int, dict[str, Any]]:
        """(page count, ``export_to_dict()`` of the Docling document); ``ConversionFailed`` on bad
        input."""
        ...


def load_converter() -> Any:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise EngineUnavailable(f"docling is not installed: {install_hint()}") from exc
    return DocumentConverter()


class DoclingConverter:
    """One ``DocumentConverter`` (its models load on the first conversion); calls are serialised."""

    def __init__(self, *, loader: Callable[[], Any] = load_converter) -> None:
        self._loader = loader
        self._converter: Any | None = None
        self._lock = threading.Lock()

    def convert(self, path: Path) -> tuple[int, dict[str, Any]]:
        with self._lock:
            if self._converter is None:
                self._converter = self._loader()
            try:
                document = self._converter.convert(str(path)).document
                return int(document.num_pages()), document.export_to_dict()
            except Exception as exc:  # docling raises many unrelated types for bad input
                raise ConversionFailed(f"docling could not convert {path.name}: {exc}") from exc
