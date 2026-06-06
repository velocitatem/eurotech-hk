"""Auto-discovers all Loader subclasses defined in this package.

Real loaders from parallel work: drop a .py file here, subclass Loader,
and they will be picked up automatically.
"""
from __future__ import annotations
import importlib, pkgutil
from pathlib import Path
from .base import Loader


def get_all_loaders() -> list[Loader]:
    pkg_path = str(Path(__file__).parent)
    pkg_name = __name__.rsplit(".", 1)[0]
    for _, mod_name, _ in pkgutil.iter_modules([pkg_path]):
        if mod_name not in ("base", "registry"):
            importlib.import_module(f"{pkg_name}.{mod_name}")
    return [cls() for cls in Loader.__subclasses__()]
