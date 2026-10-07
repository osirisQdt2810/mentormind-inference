"""POST /v1/documents/convert: Docling's document as JSON; bad files -> 422 (docling faked)."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.gateway.fakes import FakeConverter, make_config
from vlm_server.gateway.app import create_app
from vlm_server.gateway.documents import DoclingConverter

URL = "/v1/documents/convert"
PDF = b"%PDF-1.7 fake"


def post(client: TestClient, name: str = "SOP máy ép.pdf", content: bytes = PDF) -> Any:
    return client.post(URL, files={"file": (name, content, "application/octet-stream")})


def test_converted_document_shape() -> None:
    document = {"schema_name": "DoclingDocument", "texts": [{"text": "Bước 1"}]}
    converter = FakeConverter(result=(3, document))
    response = post(TestClient(create_app(make_config(), converter=converter)))
    assert response.status_code == 200
    assert response.json() == {"filename": "SOP máy ép.pdf", "num_pages": 3, "document": document}
    assert converter.calls == [("SOP máy ép.pdf", PDF)]  # docling sees the original name/suffix


def test_failed_conversion_is_422() -> None:
    converter = FakeConverter(error="docling could not convert a.pptx: corrupt zip")
    response = post(TestClient(create_app(make_config(), converter=converter)), name="a.pptx")
    assert response.status_code == 422
    assert response.json() == {"error": "docling could not convert a.pptx: corrupt zip"}


@pytest.mark.parametrize("name", ["notes.txt", "photo.png", "no-suffix"])
def test_unsupported_file_type_is_422(name: str) -> None:
    converter = FakeConverter()
    response = post(TestClient(create_app(make_config(), converter=converter)), name=name)
    assert response.status_code == 422 and "unsupported file type" in response.json()["error"]
    assert converter.calls == []


@pytest.mark.parametrize("name", ["a.PDF", "b.docx", "c.xlsx", "d.pptx"])
def test_the_four_office_formats_are_accepted(name: str) -> None:
    response = post(TestClient(create_app(make_config(), converter=FakeConverter())), name=name)
    assert response.status_code == 200 and response.json()["filename"] == name


def test_empty_upload_is_400() -> None:
    response = post(TestClient(create_app(make_config(), converter=FakeConverter())), content=b"")
    assert response.status_code == 400 and "empty" in response.json()["error"]


def test_path_in_the_filename_is_dropped() -> None:
    converter = FakeConverter()
    post(TestClient(create_app(make_config(), converter=converter)), name="../../etc/x.docx")
    assert converter.calls[0][0] == "x.docx"


def test_docling_engine_exports_the_document_and_wraps_errors() -> None:
    document = SimpleNamespace(num_pages=lambda: 4, export_to_dict=lambda: {"pages": {}})
    results = iter([SimpleNamespace(document=document), ValueError("broken xref")])

    def convert(path: str) -> Any:
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    loads: list[int] = []
    engine = DoclingConverter(loader=lambda: loads.append(1) or SimpleNamespace(convert=convert))
    client = TestClient(create_app(make_config(), converter=engine))
    assert post(client).json()["num_pages"] == 4
    failed = post(client)
    assert failed.status_code == 422 and "broken xref" in failed.json()["error"]
    assert loads == [1]  # one converter, kept


def test_missing_docling_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "docling", None)
    monkeypatch.setitem(sys.modules, "docling.document_converter", None)
    response = post(TestClient(create_app(make_config())))
    assert response.status_code == 503 and "docling" in response.json()["error"]
