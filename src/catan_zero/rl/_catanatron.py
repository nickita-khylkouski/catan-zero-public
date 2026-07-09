from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


class CatanatronUnavailableError(RuntimeError):
    """Raised when neither installed nor vendored Catanatron can be imported."""


def ensure_catanatron() -> ModuleType:
    """Import Catanatron, falling back to the repository vendor tree if present."""

    repo_root = Path(__file__).resolve().parents[3]
    vendor_path = repo_root / "vendor" / "catanatron" / "catanatron"
    vendor_path_str = str(vendor_path)
    if vendor_path.is_dir() and vendor_path_str not in sys.path:
        # The current PyPI package can be missing modules this project depends on
        # such as catanatron.features. Prefer the pinned vendored tree when present.
        sys.path.insert(0, vendor_path_str)

    try:
        return importlib.import_module("catanatron")
    except ImportError as installed_error:
        if vendor_path.is_dir():
            try:
                return importlib.import_module("catanatron")
            except ImportError as vendor_error:
                raise CatanatronUnavailableError(
                    "Catanatron is not importable. Install catanatron or keep the "
                    "vendored source tree at vendor/catanatron/catanatron."
                ) from vendor_error
        raise CatanatronUnavailableError(
            "Catanatron is not importable. Install catanatron or keep the vendored "
            "source tree at vendor/catanatron/catanatron."
        ) from installed_error


def import_catanatron_module(name: str) -> ModuleType:
    ensure_catanatron()
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as module_error:
        if not name.startswith("catanatron"):
            raise
        repo_root = Path(__file__).resolve().parents[3]
        vendor_path = repo_root / "vendor" / "catanatron" / "catanatron"
        if not vendor_path.is_dir():
            raise
        vendor_path_str = str(vendor_path)
        if vendor_path_str not in sys.path:
            sys.path.insert(0, vendor_path_str)
        for module_name in tuple(sys.modules):
            if module_name == "catanatron" or module_name.startswith("catanatron."):
                del sys.modules[module_name]
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            raise module_error
