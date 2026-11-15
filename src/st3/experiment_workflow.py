"""Shared one-run contract. No Colab imports, scientific settings or cloud side effects."""

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
import zipfile


STAGES = ("Validate", "Benchmark", "Experiment", "CV", "Verify", "Decision",
          "GuardedConfirmation", "FinalModel", "Inference", "SubmissionValidation")
CONFIGURATION = ("AUTHORIZE_EXPERIMENT", "AUTHORIZE_GUARDED_CONFIRMATION",
                 "RESOURCE_ELIGIBILITY_STATUS", "MAX_GPU_HOURS", "MAX_TOTAL_WALL_HOURS",
                 "REQUIRE_SUBMISSION_ZIP", "AUTO_BROWSER_DOWNLOAD", "AUTO_RELEASE_COLAB",
                 "EVAL_INPUT_CSV", "EVAL_DATA_DIR")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_bytes((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    temporary.replace(path)


class StopRun(RuntimeError):
    def __init__(self, status, message):
        self.status = status
        super().__init__(message)


class Budget:
    """GPU allocation occupancy, not kernel time; includes setup since owner Run all."""
    def __init__(self, gpu_hours, wall_hours, *, cuda, reserve=180, started=None, clock=time.time):
        if any(not math.isfinite(x) or x <= 0 for x in (gpu_hours, wall_hours)):
            raise ValueError("Budgets must be finite positive hours")
        self.clock, self.started = clock, clock() if started is None else started
        self.seconds = min(wall_hours * 3600, gpu_hours * 3600 if cuda else math.inf)
        self.reserve = reserve

    def remaining(self):
        return self.seconds - (self.clock() - self.started)

    def check(self, projected=0):
        if not math.isfinite(projected) or projected < 0 or self.remaining() < projected + self.reserve:
            raise StopRun("STOPPED_BUDGET", f"Remaining {self.remaining():.1f}s; need {projected:.1f}s + {self.reserve}s persistence reserve")


def safe_member(name):
    p = PurePosixPath(name)
    if not name or "\\" in name or p.is_absolute() or any(x in ("", ".", "..") for x in name.split("/")) or ":" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    return p


def restore_handoff(archive, destination, archive_sha, manifest_sha):
    """Verify the entire archive before extracting; never follows symlinks or overwrites."""
    archive, destination = Path(archive), Path(destination)
    if sha256(archive) != archive_sha:
        raise StopRun("BLOCKED_INPUT", "Handoff ZIP SHA-256 mismatch; use the delivered original ZIP")
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise StopRun("BLOCKED_INPUT", "Duplicate ZIP paths")
        for info in z.infolist():
            safe_member(info.filename)
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise StopRun("BLOCKED_INPUT", "ZIP symlinks are forbidden")
        raw = z.read("input_manifest.json")
        if hashlib.sha256(raw).hexdigest() != manifest_sha:
            raise StopRun("BLOCKED_INPUT", "Input manifest SHA-256 mismatch")
        manifest = json.loads(raw)
        if set(names) != {"input_manifest.json", *manifest["files"]}:
            raise StopRun("BLOCKED_INPUT", "ZIP file coverage differs from manifest")
        for name, digest in manifest["files"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise StopRun("BLOCKED_INPUT", f"Handoff corruption: {name}")
        if destination.exists():
            if any(not (destination / n).is_file() or sha256(destination / n) != h for n, h in manifest["files"].items()):
                raise StopRun("BLOCKED_INPUT", "Existing local inputs differ; choose a fresh input directory")
            return manifest
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
            stage = Path(temp) / "inputs"
            stage.mkdir()
            z.extractall(stage)
            stage.replace(destination)
    return manifest


def publish_model(directory, identity, writer):
    """Only complete, sealed directories become resumable. An interrupted temp is ignored."""
    directory = Path(directory)
    if directory.exists():
        return verify_model(directory, identity)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=directory.parent) as temp:
        stage = Path(temp) / "model"
        stage.mkdir()
        writer(stage)
        files = {p.relative_to(stage).as_posix(): sha256(p) for p in stage.rglob("*") if p.is_file()}
        if not files:
            raise ValueError("Cannot publish an empty model")
        write_json(stage / "seal.json", {"identity": identity, "files": files, "complete": True})
        stage.replace(directory)
    return verify_model(directory, identity)


def verify_model(directory, identity):
    directory = Path(directory)
    try:
        seal = json.loads((directory / "seal.json").read_text())
    except (OSError, ValueError) as exc:
        raise StopRun("FAILED_VERIFICATION", f"Missing/invalid completed model seal: {directory}") from exc
    if seal.get("complete") is not True or seal["identity"] != identity:
        raise StopRun("FAILED_VERIFICATION", f"Completed model identity mismatch: {directory}")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != {*seal["files"], "seal.json"}:
        raise StopRun("FAILED_VERIFICATION", f"Model file coverage changed: {directory}")
    for name, digest in seal["files"].items():
        safe_member(name)
        if sha256(directory / name) != digest:
            raise StopRun("FAILED_VERIFICATION", f"Model corruption: {name}")
    return seal


def export_artifacts(run_dir, destination, report):
    """Publish ZIP/receipts on persistent storage and verify every byte on read-back."""
    run_dir, destination = Path(run_dir), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    files = {p.relative_to(run_dir).as_posix(): sha256(p) for p in run_dir.rglob("*")
             if p.is_file() and not any(part.startswith(".pending-") for part in p.parts)
             and not p.name.endswith(".partial") and p.name not in ("SUCCESS", "outputs.zip", "outputs.sha256")
             and (p.name != "submission.zip" or report.get("status") == "SUCCESS_SUBMISSION_READY")}
    target = destination / "outputs.zip"
    with tempfile.TemporaryDirectory(dir=destination) as temp:
        staged = Path(temp) / "outputs.zip"
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as z:
            for name in files:
                z.write(run_dir / name, name)
            z.writestr("export_manifest.json", json.dumps({"files": files, "report": report}, sort_keys=True))
        # Flush and read persistent bytes; return only after complete verification.
        with staged.open("r+b") as handle:
            os.fsync(handle.fileno())
        digest = sha256(staged)
        staged.replace(target)
        if sha256(target) != digest:
            raise OSError("Persistent archive read-back SHA mismatch")
        with zipfile.ZipFile(target) as z:
            if set(z.namelist()) != {*files, "export_manifest.json"}:
                raise OSError("Export coverage mismatch")
            for name, expected in files.items():
                if hashlib.sha256(z.read(name)).hexdigest() != expected:
                    raise OSError(f"Persistent export corrupt: {name}")
    (destination / "outputs.sha256").write_text(digest + "  outputs.zip\n")
    submission = run_dir / "submission.zip"
    published = {"outputs.zip": target}
    if report.get("status") == "SUCCESS_SUBMISSION_READY" and submission.exists():
        copy = destination / "submission.zip"
        if submission.resolve() != copy.resolve():
            shutil.copyfile(submission, copy)
        if sha256(copy) != sha256(submission):
            raise OSError("Submission persistent read-back mismatch")
        published["submission.zip"] = copy
    return published


def finalize(run_dir, destination, report, *, platform, download=None, release=None,
             auto_download=True, auto_release=True, exporter=export_artifacts):
    """Cleanup covers bootstrap, science and export failures. Injectable, CPU-testable APIs."""
    run_dir, destination = Path(run_dir), Path(destination)
    attempts, published = [], {}
    report = dict(report, browser_arrival="UNVERIFIED", physical_runtime_termination="UNVERIFIED")
    def record():
        try:
            write_json(destination / "final_report.json", {**report, "downloads": attempts})
        except Exception:
            pass  # This cannot prevent the release attempt below.
    try:
        try:
            published = exporter(run_dir, destination, report)
            report["persistent_readback"] = "PASS"
        except Exception as exc:
            report.update(stage_status_before_export=report.get("status"), status="FAILED_EXPORT", error=str(exc), persistent_readback="FAIL")
        record()
        for name in ("submission.zip", "outputs.zip"):
            item = {"file": name, "arrival": "UNVERIFIED"}
            if name not in published:
                item["status"] = "SKIPPED_NO_VERIFIED_FILE"
            elif platform != "colab" or not auto_download:
                item["status"] = "SKIPPED_DISABLED_OR_PLATFORM"
            else:
                try:
                    item.update(status="REQUESTED", sha256=sha256(published[name]))
                    attempts.append(item)
                    record()
                    download(str(published[name]))
                    item["status"] = "API_RETURNED"
                except Exception as exc:
                    item.update(status="FAILED", error=str(exc))
                record()
                continue
            attempts.append(item)
    finally:
        report["runtime_release"] = "REQUESTED_NOT_CONFIRMED" if platform == "colab" and auto_release else "NOT_APPLICABLE_OR_DISABLED"
        record()
        if platform == "colab" and auto_release:
            try:
                release()
                report["runtime_release"] = "API_RETURNED"
            except Exception as exc:
                report.update(runtime_release="FAILED", runtime_release_error=str(exc))
        record()
    return report


def validate_notebook_contract(notebook):
    """Structural entry-point checks complement mandatory behavioral contract tests."""
    import ast
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    configs = [c for c in cells if "# OWNER_CONFIGURATION" in c]
    if len(configs) != 1:
        raise ValueError("Exactly one owner configuration section is required")
    tree = ast.parse(configs[0])
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    if not set(CONFIGURATION).issubset(names):
        raise ValueError("Missing required workflow configuration")
    if sum("run_cloud(" in c for c in cells) != 1:
        raise ValueError("One orchestration call is required")
    if any("input(" in c for c in cells):
        raise ValueError("No intermediate interactive prompts")
    for cell in cells:
        ast.parse(cell)
    return {"status": "PASS", "boundary": "Structure only; behavioral CPU compliance tests are also required"}


def execute_stages(callbacks, budget, progress):
    """Reusable fail-closed transitions. Callers supply science, never manual Continue."""
    result = {}
    for stage in STAGES:
        budget.check()
        progress(stage)
        result[stage] = callbacks[stage]()
        if stage == "Verify" and result[stage].get("status") != "PASS":
            raise StopRun("FAILED_VERIFICATION", "Independent CV replay failed")
        if stage == "Decision" and result[stage].get("status") != "CV_GO":
            raise StopRun("CV_NO_GO", "Frozen scientific CV gate failed")
        if stage == "GuardedConfirmation" and result[stage].get("status") != "CONFIRMATION_GO":
            raise StopRun("CONFIRMATION_NO_GO", "Frozen confirmation gate failed")
        if stage == "FinalModel" and result[stage].get("status") != "PASS":
            raise StopRun("FAILED_VERIFICATION", "Final model identities are not verified")
        if stage == "SubmissionValidation" and result[stage].get("status") != "PASS":
            raise StopRun("FAILED_SUBMISSION", "Submission coverage/ZIP validation failed")
    return result
