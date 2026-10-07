"""Errors the gateway turns into ``{"error": "..."}`` responses with their status code."""

from __future__ import annotations


class GatewayError(Exception):
    status_code = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class BadRequest(GatewayError):
    """The request itself is wrong (empty upload, unsupported option, undecodable audio)."""

    status_code = 400


class ConversionFailed(GatewayError):
    """Docling could not convert the uploaded document (unsupported or broken file)."""

    status_code = 422


class UpstreamUnavailable(GatewayError):
    """The VLM upstream did not answer."""

    status_code = 502


class EngineUnavailable(GatewayError):
    """The engine's library is not installed, or its model could not be loaded."""

    status_code = 503
