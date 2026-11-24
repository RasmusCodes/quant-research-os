"""Execution-environment and code-version capture.

Recorded with every experiment: interpreter, platform, key package versions, a hash of the full
installed-package list, and the git state of the repository that holds the experiment's code.
Deliberately NOT recorded: host name, user name and absolute paths (privacy and portability).
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import inspect
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from qros._version import __version__
from qros.util.hashing import sha256_file, sha256_text

KEY_PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "torch", "statsmodels")


def installed_packages() -> list[str]:
    names = set()
    for dist in md.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:  # pragma: no cover - broken metadata
            continue
        if name:
            names.add(f"{name.lower()}=={dist.version}")
    return sorted(names)


def capture_environment(extra_packages: tuple[str, ...] = ()) -> dict[str, Any]:
    packages = {}
    for name in KEY_PACKAGES + tuple(extra_packages):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            continue
    all_packages = installed_packages()
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "byteorder": sys.byteorder,
        "qros": __version__,
        "packages": packages,
        "packages_count": len(all_packages),
        "packages_sha256": sha256_text("\n".join(all_packages)),
    }


_CRED = re.compile(r"(https?://)[^/@\s]+@")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info(path: str | Path | None = None) -> dict[str, Any] | None:
    """Commit, branch, dirty flag and origin URL (credentials stripped) of the repo holding ``path``."""
    start = Path(path or os.getcwd())
    cwd = start if start.is_dir() else start.parent
    top = _git(["rev-parse", "--show-toplevel"], cwd)
    if not top:
        return None
    commit = _git(["rev-parse", "HEAD"], cwd)
    status = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    remote = _git(["remote", "get-url", "origin"], cwd)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
        "remote": _CRED.sub(r"\1", remote) if remote else None,
        "repo": Path(top).name,
    }


def resolve_entry_point(entry_point: str) -> Callable[..., Any]:
    """Import ``package.module:function`` and return the callable."""
    if ":" not in entry_point:
        raise ValueError(f"Entry point must look like 'module:function', got {entry_point!r}")
    module_name, attr = entry_point.split(":", 1)
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attr.split("."):
        target = getattr(target, part)
    if not callable(target):
        raise TypeError(f"{entry_point} is not callable")
    return target


def code_fingerprint(func_or_module: Any) -> dict[str, Any]:
    """Hash of the source file that defines a function/module, plus the repo's git state."""
    module = inspect.getmodule(func_or_module) or func_or_module
    try:
        source_file = inspect.getsourcefile(module)
    except TypeError:
        source_file = None
    info: dict[str, Any] = {
        "module": getattr(module, "__name__", None),
        "qualname": getattr(func_or_module, "__qualname__", None),
    }
    if source_file and Path(source_file).is_file():
        info["file"] = Path(source_file).name
        info["file_sha256"] = sha256_file(source_file)
        git = git_info(source_file)
        if git:
            info["git"] = git
    pkg = (info.get("module") or "").split(".")[0]
    if pkg:
        try:
            info["package_version"] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return info
