"""Runtime helpers for the locally installed Phase 7 model libraries."""
from __future__ import annotations

import sys
from pathlib import Path


def expose_existing_base_site_packages() -> Path | None:
    """Expose an already installed base-Python package directory when running in a venv.

    The project environment intentionally stays isolated.  On the supplied Windows
    machine, the existing Torch/Transformers stack is installed in the base Python
    3.11 site-packages, so add that directory only as an import fallback.  This
    function never installs packages or accesses the network.
    """
    candidates = [
        Path(sys.base_prefix) / "Lib" / "site-packages",
        Path(sys.base_prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any((candidate / package).exists() for package in ("torch", "transformers", "sentence_transformers")):
            value = str(candidate)
            if value not in sys.path:
                sys.path.append(value)
            return candidate
    return None
