"""Safe, explicit JSON/YAML configuration loaders for ZS32 contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from zs32_inspection.domain.contracts import DeploymentContract, InspectionRecipe, ReleaseManifest, RoiConfig
from zs32_inspection.domain.errors import ConfigLoadError
from zs32_inspection.domain.topology import CaptureTopology

from .schemas import (
    parse_deployment_contract,
    parse_recipe,
    parse_release_manifest,
    parse_roi_config,
    parse_topology,
)


def load_config_object(path: str | Path) -> dict[str, Any]:
    """Safely load one explicitly named JSON or YAML object."""
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    if suffix not in {".json", ".yaml", ".yml"}:
        msg = f"configuration must use .json, .yaml, or .yml: {config_path}"
        raise ConfigLoadError(msg)
    try:
        text = config_path.read_text(encoding="utf-8")
        payload = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as error:
        msg = f"cannot load configuration {config_path}: {error}"
        raise ConfigLoadError(msg) from error
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        msg = f"configuration root must be an object: {config_path}"
        raise ConfigLoadError(msg)
    return payload


def load_topology(path: str | Path) -> CaptureTopology:
    """Load and validate a topology JSON/YAML file."""
    return parse_topology(load_config_object(path))


def load_roi_config(path: str | Path) -> RoiConfig:
    """Load and validate the authoritative hand-aware ROI JSON/YAML file."""
    return parse_roi_config(load_config_object(path))


def load_recipe(path: str | Path) -> InspectionRecipe:
    """Load and validate a deployment recipe JSON/YAML file."""
    return parse_recipe(load_config_object(path))


def load_deployment_contract(path: str | Path) -> DeploymentContract:
    """Load, rehash, and exhaustively validate a serialized deployment contract."""
    return parse_deployment_contract(load_config_object(path))


def load_release_manifest(path: str | Path) -> ReleaseManifest:
    """Load and strictly validate a non-cyclic release manifest."""
    return parse_release_manifest(load_config_object(path))
