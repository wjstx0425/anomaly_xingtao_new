"""Typed capture failures used for fail-closed status classification."""

from __future__ import annotations


class CaptureFailure(RuntimeError):
    """Base class for failures that have an explicit inspection status."""

    failure_kind: str


class InvalidCaptureError(CaptureFailure):
    """The acquired capture is incomplete or conflicts with its identity contract."""

    failure_kind = "INVALID_CAPTURE"


class CaptureDataIntegrityError(InvalidCaptureError):
    """A persisted capture is missing data or violates a verified identity contract."""


class CaptureRetakeRequired(CaptureFailure):
    """Quality or registration measurements require a fresh acquisition."""

    failure_kind = "RETAKE"


class CaptureSystemError(CaptureFailure):
    """A source, gate, configuration, filesystem, or publication runtime failed."""

    failure_kind = "SYSTEM_ERROR"
