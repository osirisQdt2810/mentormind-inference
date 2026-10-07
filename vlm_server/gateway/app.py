"""The gateway's FastAPI app: ASR, embeddings and documents on the CPU, the VLM proxied.

    GET  /health                    status, VLM upstream, model of each service
    POST /v1/audio/transcriptions   multipart ``file`` -> verbose_json with word timestamps
    POST /v1/embeddings             {"input": str | [str]} -> L2-normalised bge-m3 vectors
    POST /v1/documents/convert      multipart ``file`` -> Docling document as JSON
    *    /v1/*                      anything else -> the VLM upstream (vLLM), streamed back

No authentication here: the Caddy edge in front checks the bearer token. Errors are
``{"error": "..."}``: 400/422 bad input, 502 VLM upstream down, 503 engine library missing.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from vlm_server.gateway.asr import AsrEngine, FasterWhisperAsr, Transcription
from vlm_server.gateway.config import GatewayConfig
from vlm_server.gateway.documents import (
    SUFFIXES,
    ConvertedDocument,
    DoclingConverter,
    DocumentEngine,
)
from vlm_server.gateway.embeddings import (
    BgeM3Embedder,
    Embedder,
    EmbeddingItem,
    EmbeddingRequest,
    EmbeddingResponse,
    Usage,
    l2_normalise,
)
from vlm_server.gateway.errors import BadRequest, ConversionFailed, GatewayError
from vlm_server.gateway.proxy import forward

PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
#: Seconds to wait for a TCP connection to the VLM; the request itself may take proxy_timeout_s.
CONNECT_TIMEOUT_S = 10.0


def create_app(
    config: GatewayConfig | None = None,
    *,
    asr: AsrEngine | None = None,
    embedder: Embedder | None = None,
    converter: DocumentEngine | None = None,
    http: httpx.AsyncClient | None = None,
) -> FastAPI:
    """The app; each engine (and the proxy's HTTP client) can be injected, e.g. fakes in tests."""
    cfg = config or GatewayConfig()
    asr_engine: AsrEngine = asr or FasterWhisperAsr.from_config(cfg)
    embed_engine: Embedder = embedder or BgeM3Embedder(cfg.embed_model, batch_size=cfg.embed_batch)
    doc_engine: DocumentEngine = converter or DoclingConverter()
    client = http or httpx.AsyncClient(
        timeout=httpx.Timeout(
            cfg.proxy_timeout_s, connect=min(cfg.proxy_timeout_s, CONNECT_TIMEOUT_S)
        ),
        trust_env=False,  # the upstream is local: never route it through an HTTP(S)_PROXY
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if http is None:
            await client.aclose()

    app = FastAPI(title="mentormind-inference gateway", lifespan=lifespan)

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse({"error": exc.message}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('msg', '')}"
            for error in exc.errors()
        )
        return JSONResponse({"error": f"invalid request: {problems}"}, status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "vlm_upstream": cfg.vlm_upstream,
            "services": {
                "asr": cfg.asr_model,
                "embeddings": cfg.embed_model,
                "documents": "docling",
            },
        }

    @app.post("/v1/audio/transcriptions")
    def transcriptions(
        file: Annotated[UploadFile, File()],
        model: Annotated[str | None, Form()] = None,  # accepted; one ASR model is served
        language: Annotated[str | None, Form()] = None,
        response_format: Annotated[str | None, Form()] = None,
    ) -> Transcription:
        if (response_format or "verbose_json") != "verbose_json":
            raise BadRequest(f"response_format {response_format!r}: only verbose_json is supported")
        with saved_upload(file) as path:
            return asr_engine.transcribe(path, language=_language(language))

    @app.post("/v1/embeddings")
    def embeddings(body: EmbeddingRequest) -> EmbeddingResponse:
        texts = [body.input] if isinstance(body.input, str) else body.input
        if not texts:
            raise BadRequest("input must not be empty")
        vectors, tokens = embed_engine.encode(texts)
        if len(vectors) != len(texts):
            raise GatewayError(f"embedder returned {len(vectors)} vectors for {len(texts)} texts")
        return EmbeddingResponse(
            model=cfg.embed_model,
            data=[EmbeddingItem(index=i, embedding=l2_normalise(v)) for i, v in enumerate(vectors)],
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
        )

    @app.post("/v1/documents/convert")
    def convert_document(file: Annotated[UploadFile, File()]) -> ConvertedDocument:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUFFIXES:
            raise ConversionFailed(
                f"unsupported file type {suffix or '(none)'!r}: send {', '.join(SUFFIXES)}"
            )
        with saved_upload(file) as path:
            num_pages, document = doc_engine.convert(path)
        return ConvertedDocument(
            filename=file.filename or path.name, num_pages=num_pages, document=document
        )

    @app.api_route("/v1/{path:path}", methods=PROXY_METHODS, include_in_schema=False)
    async def vlm(request: Request, path: str) -> Response:
        return await forward(request, client, cfg.vlm_upstream, heartbeat_s=cfg.heartbeat_s)

    return app


def _language(value: str | None) -> str | None:
    """Empty, missing or "auto" = let Whisper detect it."""
    value = (value or "").strip()
    return None if value.lower() in ("", "auto") else value


@contextmanager
def saved_upload(upload: UploadFile) -> Iterator[Path]:
    """The upload as a file under its own (sanitised) name in a temporary directory.

    The name keeps its suffix: Docling and ffmpeg pick the format from it.
    """
    name = Path(upload.filename or "").name
    suffix = Path(name).suffix
    if name in ("", ".", "..") or len(name.encode()) > 200:
        name = "upload" + (suffix if len(suffix.encode()) <= 20 else "")
    with tempfile.TemporaryDirectory(prefix="mentormind-gateway-") as tmp:
        path = Path(tmp) / name
        with path.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        if path.stat().st_size == 0:
            raise BadRequest(f"{name}: the uploaded file is empty")
        yield path
