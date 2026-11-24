"""Integration with other repositories: run manifests, project adapters and quick registration."""

from __future__ import annotations

from typing import Any

from qros.adapters.manifest import SCHEMA_ID, ManifestError, ingest_manifest, load_manifest, validate_manifest
from qros.adapters.projects import ADAPTERS, ProjectAdapter, get_adapter


def register_returns(registry: Any, name: str, returns: Any, **kwargs: Any) -> Any:
    """One call for Python projects: track an experiment and log its return series (or result)."""
    from qros.tracking import track
    validate = kwargs.pop("validate", True)
    run = track(name, registry=registry, **kwargs)
    with run:
        return run.log_result(returns, validate=validate)


__all__ = [
    "SCHEMA_ID", "ManifestError", "ingest_manifest", "load_manifest", "validate_manifest", "ADAPTERS",
    "ProjectAdapter", "get_adapter", "register_returns",
]
