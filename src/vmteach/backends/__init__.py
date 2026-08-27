"""Backend registry — auto-discovers installed backends.

Each backend is a subdirectory with an __init__.py exposing:
  - create_sim(**params) -> SomeSimClass
  - setup_<name>(**params) -> (core, sim)

Programmatic use:
    from vmteach.backends import load_backend, list_backends
    core, sim = load_backend("bacteria", n_cells=50, seed=0)
    print(list_backends())
"""

from __future__ import annotations
import importlib
from pathlib import Path


def list_backends() -> list[str]:
    """Return sorted list of available backend names."""
    d = Path(__file__).parent
    return sorted(
        p.name for p in d.iterdir()
        if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith("_")
    )


def load_backend(name: str, **kwargs):
    """Load a backend by name and call its setup function.

    Args:
        name: Backend name (e.g. "bacteria", "voronoi", "zebrafish")
        **kwargs: Forwarded to the backend's setup_<name>() function.

    Returns:
        (core, sim) tuple — or just core for the particle backend.
    """
    mod = importlib.import_module(f"vmteach.backends.{name}")
    fn_name = f"setup_{name}"
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise AttributeError(
            f"Backend '{name}' has no setup function "
            f"(expected setup_{name})"
        )
    return fn(**kwargs)


def describe_backend(name: str) -> dict:
    """Return BACKEND_INFO metadata for a single backend."""
    mod = importlib.import_module(f"vmteach.backends.{name}")
    return getattr(mod, "BACKEND_INFO", {})


def describe_backends() -> dict[str, dict]:
    """Return BACKEND_INFO for all discovered backends."""
    return {name: describe_backend(name) for name in list_backends()}
