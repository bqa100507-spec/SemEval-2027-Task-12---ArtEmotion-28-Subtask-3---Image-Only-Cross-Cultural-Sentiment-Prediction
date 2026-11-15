"""Checksum-verified notebook bootstrap. Executed only from the pinned source archive."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile


def run_cloud(settings, *, platform, mount, download, release, started):
    # Notebook already checks ZIP/manifest/source/entry hashes before executing us.
    # Check again before making any helper import executable.
    archive = Path(settings["INPUT_ZIP"])
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != settings["ZIP_SHA256"]:
        raise ValueError("Input ZIP changed after notebook validation")
    with zipfile.ZipFile(archive) as z:
        manifest_raw = z.read("input_manifest.json")
        if hashlib.sha256(manifest_raw).hexdigest() != settings["MANIFEST_SHA256"]:
            raise ValueError("Input manifest changed")
        manifest = json.loads(manifest_raw)
        source_bytes = z.read("source/source.zip")
    if hashlib.sha256(source_bytes).hexdigest() != manifest["files"]["source/source.zip"]:
        raise ValueError("Source archive checksum mismatch")
    import io
    bootstrap = Path(settings["LOCAL_ROOT"]) / "trusted_bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as z:
        for name in ("experiment_workflow", "exp023_cloud"):
            raw = z.read(f"src/st3/{name}.py")
            if hashlib.sha256(raw).hexdigest() != manifest["source_files_sha256"][f"src/st3/{name}.py"]:
                raise ValueError("Cloud helper checksum mismatch")
            (bootstrap / f"{name}.py").write_bytes(raw)
    # Private package namespace avoids reusing a stale st3 module from an earlier cell.
    package = "_exp023_pinned_bootstrap"
    import types
    module = types.ModuleType(package)
    module.__path__ = [str(bootstrap)]
    sys.modules[package] = module
    for name in ("experiment_workflow", "exp023_cloud"):
        full = package + "." + name
        spec = importlib.util.spec_from_file_location(full, bootstrap / f"{name}.py")
        imported = importlib.util.module_from_spec(spec)
        sys.modules[full] = imported
        spec.loader.exec_module(imported)
    return sys.modules[package + ".exp023_cloud"].run_cloud(settings, platform=platform, mount=mount,
                                                            download=download, release=release, started=started)
