"""Prepare and verify the existing EXP-028 fold-3 experiment in an isolated CPU runtime."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BASE = "f34dbe6879ab339ddf7d0f52e1af72c1437beff9"
ZIP_SHA256 = "2db4a53679940c7eb63c7c8272b91be8ec0618bb6c04a69ebaa15909e20f64e0"
MANIFEST_SHA256 = "b5315db3847c29c432e91510a3d8f99c4b1bf13aca8b17282a53059347a20aa9"
SCIENCE = (
    "reports/exp028_design_proposal.md", "docs/exp028_design_spec.json",
    "docs/exp028_feature_inventory.json", "configs/exp028_fold3_manifest.json",
    "src/st3/exp028_cache.py", "src/st3/exp028_stage1.py",
)
SOURCE_KEYS = {"CLIP-B": "clip", "CLIP-L": "large", "SigLIP2-Base": "siglip"}


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def json_write(path, value):
    pending = path.with_suffix(path.suffix + ".part")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def pinned_source(revision, root=ROOT):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    if head != revision or dirty:
        raise ValueError("Use the clean notebook-pinned Git revision")
    for name in SCIENCE:
        original = subprocess.check_output(["git", "show", f"{BASE}:{name}"], cwd=root)
        if original != (root / name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"Frozen TASK-033 scientific source differs: {name}")
    return {name: file_hash(root / name) for name in (
        *SCIENCE, "scripts/run_exp028_stage1.py", "scripts/run_exp028_colab_cpu.py",
        "requirements-exp028-cpu.lock.txt",
    )}


def restore_member(archive, files, name, destination):
    if name not in files:
        raise ValueError(f"Pinned input manifest lacks {name}")
    target = (destination / name).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError("Unexpected archive path")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix + ".part")
    with archive.open(name) as source, pending.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    if file_hash(pending) != files[name]:
        raise ValueError(f"Handoff member SHA-256 differs: {name}")
    pending.replace(target)


def write_original_npz(path, names, features):
    """Serialize the original Windows NPZ headers identically on Linux."""
    import numpy as np

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as output:
        for key, values in (("names", names), ("features", features)):
            info = zipfile.ZipInfo(f"{key}.npy")
            info.create_system = 0
            with output.open(info, "w", force_zip64=True) as handle:
                np.lib.format.write_array(handle, values, allow_pickle=False)


def rebuild_visual(inputs, workspace, rows, inventory):
    import numpy as np
    import pandas as pd
    from st3.exp028_cache import visual_blocks

    provenance = json_read(inputs / "visual/provenance.json")
    image_sha = json_read(inputs / "visual/image_hashes.json")
    p2 = rows[["image", "image_sha256"]].drop_duplicates()
    if any(image_sha.get(image) != digest for image, digest in p2.itertuples(index=False, name=None)):
        raise ValueError("Packed visual image digests differ from the P2 manifest")
    source_encoders = {"clip": provenance["encoders"]["clip"], "large": provenance["clip_large"],
                       "siglip": provenance["encoders"]["siglip2"]}
    protocol_images = sorted(pd.read_csv(inputs / "training/protocol.csv", usecols=["image"]).image)
    receipt = {}
    with np.load(inputs / "visual/features.npz", allow_pickle=False) as packed:
        names = packed["names"]
        index = {name: i for i, name in enumerate(names)}
        if len(index) != len(names) or set(packed.files) != {"names", *source_encoders}:
            raise ValueError("Packed visual archive contains unexpected keys")
        for entry in inventory["visual_caches"]:
            model = entry["model"]
            key = SOURCE_KEYS[model]
            expected = entry["identity"]
            actual = source_encoders[key]
            feature = expected.get("feature", expected.get("feature_format"))
            if ((actual["model"], actual["revision"], actual["feature"], actual["width"])
                    != (expected["model"], expected["revision"], feature, entry["width"])
                    or ("assets_sha256" in expected and actual["assets_sha256"] != expected["assets_sha256"])):
                raise ValueError(f"Packed visual encoder identity differs: {model}")
            pick = [index[name] for name in protocol_images] if key == "large" else slice(None)
            target = workspace / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            write_original_npz(target, names[pick], packed[key][pick])
            digest = file_hash(target)
            if digest != entry["sha256"]:
                raise ValueError(f"Reconstructed {model} NPZ differs from the frozen TASK-033 SHA-256")
            receipt[model] = {"path": entry["path"], "sha256": digest, "revision": actual["revision"]}
    # The existing loader checks name coverage, finite values, dtype, widths and norms.
    checked = visual_blocks(ROOT, workspace, rows)
    for model, vectors in checked.items():
        receipt[model]["P2_images_verified"] = len(vectors)
    return receipt


def prepare_inputs(handoff, inputs, workspace):
    from st3.exp028_cache import read_protocol

    if file_hash(handoff) != ZIP_SHA256:
        raise ValueError("Input ZIP SHA-256 differs. Expected the unchanged exp023_handoff.zip")
    core = ("training/pairs.csv", "training/protocol.csv", "history/exp022_R0.csv",
            "visual/features.npz", "visual/provenance.json", "visual/image_hashes.json")
    with zipfile.ZipFile(handoff) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate input ZIP members")
        raw = archive.read("input_manifest.json")
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
            raise ValueError("Original handoff manifest SHA-256 differs")
        pins = json.loads(raw)["files"]
        for name in core:
            restore_member(archive, pins, name, inputs)
        rows = read_protocol(ROOT, inputs)
        pictures = rows[["image", "image_sha256"]].drop_duplicates()
        for image, digest in pictures.itertuples(index=False, name=None):
            name = f"artwork/images/{image}"
            if pins.get(name) != digest:
                raise ValueError("Pinned P2 artwork SHA-256 differs")
            restore_member(archive, pins, name, inputs)
    visuals = rebuild_visual(inputs, workspace, rows, json_read(ROOT / "docs/exp028_feature_inventory.json"))
    return {"input_zip_sha256": ZIP_SHA256, "input_manifest_sha256": MANIFEST_SHA256,
            "restored_file_sha256": {k: pins[k] for k in core}, "P2_artwork_files": len(pictures),
            "visual_caches": visuals}


def dependency_receipt():
    if platform.python_version() != "3.12.10":
        raise RuntimeError("Expected the pinned Python 3.12.10 runtime")
    versions = {}
    lock = ROOT / "requirements-exp028-cpu.lock.txt"
    for line in lock.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, pin = line.split("==")
        version = importlib.metadata.version(name)
        if version != pin:
            raise RuntimeError(f"CPU dependency {name}=={pin} required, got {version}")
        versions[name] = version
    return versions


def memory_report(refit_rows):
    import psutil

    memory = psutil.virtual_memory()
    # The frozen worker reserves 6 times the raw CSR storage before fitting.
    # Budget additional headroom for arrays, PCA temporaries and Python.
    max_width = 2048
    csr_bytes = refit_rows * (2 * max_width + 1) * (8 + 4) + (refit_rows + 1) * 4
    required = 6 * csr_bytes + 1024**3
    cgroup_remaining = None
    cap, use = (Path("/sys/fs/cgroup") / k for k in ("memory.max", "memory.current"))
    if cap.is_file() and use.is_file():
        raw = cap.read_text().strip()
        if raw != "max":
            cgroup_remaining = max(0, int(raw) - int(use.read_text()))
    available = min(memory.available, cgroup_remaining) if cgroup_remaining is not None else memory.available
    return {"physical_available_bytes": memory.available, "cgroup_remaining_bytes": cgroup_remaining,
            "effective_available_bytes": available, "required_available_bytes": required,
            "raw_CSR_upper_bound_bytes": csr_bytes, "pass": available >= required}


def preflight_cache(qwen_dir, rows, receipt):
    from st3.exp028_cache import expected_identity

    if not qwen_dir.is_dir():
        raise FileNotFoundError(f"Expected the original Qwen cache directory at {qwen_dir}")
    if json_read(qwen_dir / "identity.json") != expected_identity(ROOT):
        raise ValueError("Qwen cache identity differs from the frozen extractor")
    # The scientific worker verifies the NPZ payloads during its single real audit.
    # Require every vision file too, since the frozen worker's fit gate is keyed on decoder rows.
    files = {p.name for p in qwen_dir.iterdir() if p.is_file()}
    required = {f"{digest}_{kind}.npz" for digest in rows.image_sha256.unique() for kind in ("states", "vision")}
    missing = required - files
    json_write(receipt, {"required_NPZs": len(required), "missing_NPZs": sorted(missing)})
    if missing:
        raise ValueError(f"Original Qwen cache lacks {len(missing)} required files. First missing: {min(missing)}")


def guarded_worker(inputs, workspace, qwen, output, attempt, seconds, audit_only):
    import psutil

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONNOUSERSITE="1",
               OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", OMP_NUM_THREADS="4",
               NUMEXPR_NUM_THREADS="4", BLIS_NUM_THREADS="4")
    env.pop("PYTHONPATH", None)
    cli = [sys.executable, "-u", str(ROOT / "scripts/run_exp028_stage1.py"),
           "--stage", "audit" if audit_only else "screen", "--inputs", str(inputs),
           "--workspace", str(workspace), "--qwen-cache", str(qwen), "--run-dir", str(output),
           "--require-full-vision-cache"]
    started = time.monotonic()
    sampled_peak = 0
    kernel_peak = None
    with (attempt / "worker.log").open("w") as log, (attempt / "rss_samples.jsonl").open("w") as telemetry:
        worker = subprocess.Popen(cli, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(worker.pid)
        try:
            while worker.poll() is None:
                elapsed = time.monotonic() - started
                try:
                    rss = process.memory_info().rss
                    sampled_peak = max(sampled_peak, rss)
                    telemetry.write(json.dumps({"seconds": elapsed, "rss_bytes": rss}) + "\n")
                    telemetry.flush()
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
                if elapsed >= seconds:
                    raise TimeoutError("CPU worker exceeded its time limit")
                try:
                    worker.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    continue
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
            if sys.platform.startswith("linux"):
                import resource
                kernel_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
            json_write(attempt / "worker_resources.json", {
                "elapsed_seconds": time.monotonic() - started, "exit_code": worker.returncode,
                "sampled_peak_RSS_bytes": sampled_peak, "peak_RSS_bytes": kernel_peak,
                "peak_RSS_method": "Linux child-process high-water mark" if kernel_peak is not None else "unavailable",
            })
    return worker.returncode


def verify_screen(run, inputs):
    import numpy as np
    import pandas as pd
    from st3.exp028_cache import KEYS, partition, read_protocol
    from st3.exp028_stage1 import ARMS, evaluate, select_c

    spec = json_read(ROOT / "docs/exp028_design_spec.json")
    models = (*ARMS, "E50+L")
    report = json_read(run / "status.json")
    expected_files = {"manifest.json", "cache_audit.json", "missing_pairs.json", "evaluation.json", "fits.json"}
    expected_files |= {f"{name}_{suffix}.csv" for name in models for suffix in ("predictions", "paintings")}
    if report["status"] != "FOLD3_SCREEN_COMPLETE / EXPLORATORY" or set(report["files_sha256"]) != expected_files:
        raise ValueError("Missing complete frozen worker receipts or artifacts")
    for name, expected in report["files_sha256"].items():
        if file_hash(run / name) != expected:
            raise ValueError(f"Frozen worker artifact SHA-256 differs: {name}")

    _, _, _, held = partition(read_protocol(ROOT, inputs, labels=True), spec)
    predictions = {}
    for model in models:
        path = run / f"{model}_predictions.csv"
        pred = pd.read_csv(path, float_precision="round_trip", dtype={"painting": str, "language": str})
        if (pred.columns.tolist() != [*KEYS, "p_positive"]
                or not pred[KEYS].reset_index(drop=True).equals(held[KEYS].reset_index(drop=True))
                or not np.array_equal(pred.p_positive, pred.p_positive.round(6))):
            raise ValueError(f"Unexpected prediction keys or rounding: {model}")
        predictions[model] = pred
    historical = pd.read_csv(inputs / "history/exp022_R0.csv", float_precision="round_trip")
    reference = historical.loc[historical.fold.eq(3), [*KEYS, "p_positive"]].sort_values(KEYS).reset_index(drop=True)
    if not predictions["E50+L"].equals(reference):
        raise ValueError("Saved matched E50+L does not exactly match the frozen six-decimal reference")
    recomputed, paintings = evaluate(held, predictions, spec)
    if recomputed != json_read(run / "evaluation.json"):
        raise ValueError("Saved metrics or paired bootstrap do not reproduce")
    for model, frame in paintings.items():
        saved = pd.read_csv(run / f"{model}_paintings.csv", index_col="painting", float_precision="round_trip")
        pd.testing.assert_frame_equal(saved, frame, check_exact=True)

    fits = json_read(run / "fits.json")
    if set(fits) != {*ARMS, "R0_inner", "R0_refit"}:
        raise ValueError("Incomplete fits receipt")
    records = []
    for name in ARMS:
        fit = fits[name]
        if ([r["C"] for r in fit["selection"]] != spec["LR"]["C_grid"]
                or fit["selected_C"] != select_c(fit["selection"]) or fit["refit"]["C"] != fit["selected_C"]):
            raise ValueError(f"Unexpected C selection or refit: {name}")
        records.extend([*fit["selection"], fit["refit"]])
    for name in ("R0_inner", "R0_refit"):
        if set(fits[name]) != set(SOURCE_KEYS) or any(r["C"] != 0.3 for r in fits[name].values()):
            raise ValueError(f"Unexpected reference component fits: {name}")
        records.extend(fits[name].values())
    if any(not r["converged"] or not 0 < r["iterations"] < 3000 or r["objective_parity_abs"] > 1e-12 for r in records):
        raise ValueError("LR convergence or numerical parity gate failed")
    return recomputed


def compact_table(evaluation):
    from st3.exp028_stage1 import ARMS

    metrics = evaluation["metrics"]
    ref = metrics["E50+L"]["cultural_contrast"]
    visual_name = evaluation["strongest_visual"]
    visual = metrics[visual_name]["cultural_contrast"]
    lines = [
        "| Model | Contrast | Macro Log Loss (bits) | Δ vs E50+L | Δ vs best visual |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in (*ARMS, "E50+L"):
        c, loss = metrics[name]["cultural_contrast"], metrics[name]["log_loss"]
        lines.append(f"| {name} | {c:.6f} | {loss:.6f} | {c-ref:+.6f} | {c-visual:+.6f} |")
    lines += ["", f"Strongest visual-only control: {visual_name}.", "", "Paired painting bootstrap (10,000 draws):", ""]
    for name in ("E50+L", visual_name, "C512_shuffled_Q"):
        pair = evaluation["paired"][f"C512 vs {name}"]
        lo, hi = pair["ci95"]
        lines.append(f"- C512 vs {name}: {pair['delta']:+.6f} [{lo:+.6f}, {hi:+.6f}], "
                     f"{pair['wins']} wins / {pair['losses']} losses.")
    return "\n".join(lines) + "\n"


def run(args):
    from st3.exp028_cache import read_protocol

    attempt = args.attempt
    stage = "SOURCE"
    started = time.monotonic()
    blocks = args.scratch / "fold3"
    result = {"status": "BLOCKED_VERIFICATION", "mode": "audit" if args.audit_only else "screen",
              "source_revision": args.revision, "device": "CPU"}
    try:
        result["source_sha256"] = pinned_source(args.revision)
        stage = "DEPENDENCIES"
        result["dependencies"] = dependency_receipt()
        stage = "VERIFY_INPUTS"
        args.scratch.mkdir(parents=True, exist_ok=False)
        inputs, workspace = args.scratch / "inputs", args.scratch / "workspace"
        inputs.mkdir()
        json_write(attempt / "input_verification.json", prepare_inputs(args.handoff, inputs, workspace))
        stage = "AUDIT_CACHE_KEYS"
        preflight_cache(args.qwen_cache, read_protocol(ROOT, inputs), attempt / "cache_preflight.json")
        stage = "RAM_PREFLIGHT"
        refit_count = json_read(ROOT / "configs/exp028_fold3_manifest.json")["populations"]["refit"]["pairs"]
        memory = memory_report(refit_count)
        json_write(attempt / "ram_preflight.json", memory)
        if not args.audit_only and not memory["pass"]:
            raise MemoryError("Insufficient available RAM for the frozen screen. See ram_preflight.json and use a CPU High-RAM runtime.")
        stage = "FROZEN_WORKER"
        remaining = args.seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("No time remains for the frozen worker")
        exit_code = guarded_worker(inputs, workspace, args.qwen_cache, blocks, attempt, remaining, args.audit_only)
        if exit_code:
            problem = json_read(blocks / "status.json") if (blocks / "status.json").is_file() else {}
            raise RuntimeError(problem.get("error", problem.get("status", f"Worker exited {exit_code}. See worker.log.")))
        stage = "VERIFY"
        audit = json_read(blocks / "cache_audit.json")
        if (not audit["complete"] or audit["invalid_entries"] or audit["verified_vision_images"] != 1335
                or audit["verified_language_states"] != 13600 or audit["pixel_files_verified"] != 1335):
            raise ValueError("Incomplete Qwen or artwork audit")
        expected_audit_hash = json_read(blocks / "status.json")["cache_audit_sha256"]
        if file_hash(blocks / "cache_audit.json") != expected_audit_hash:
            raise ValueError("Qwen audit SHA-256 differs from worker receipt")
        if args.audit_only:
            result["status"] = "COMPLETE_CACHE_VERIFIED"
        else:
            evaluation = verify_screen(blocks, inputs)
            resources = json_read(attempt / "worker_resources.json")
            summary = compact_table(evaluation)
            summary += (f"\nWorker time: {resources['elapsed_seconds']:.2f}s. "
                        f"Peak RSS: {resources['peak_RSS_bytes'] or resources['sampled_peak_RSS_bytes']:,} bytes.\n")
        stage = "PERSIST"
        temporary = attempt / ".fold3_incomplete"
        shutil.copytree(blocks, temporary)
        for path in blocks.iterdir():
            if file_hash(path) != file_hash(temporary / path.name):
                raise ValueError(f"Persistent file read-back failed: {path.name}")
        temporary.replace(attempt / "fold3")
        if not args.audit_only:
            (attempt / "summary.md").write_text(summary, encoding="utf-8")
            result["status"] = "FOLD3_SCREEN_COMPLETE / EXPLORATORY"
        result["cache_audit_sha256"] = expected_audit_hash
        code = 0
    except Exception as error:
        result.update(status="BLOCKED_VERIFICATION", failed_stage=stage, error=f"{type(error).__name__}: {error}")
        if blocks.is_dir():
            diagnostic = attempt / "failed_worker"
            diagnostic.mkdir(exist_ok=True)
            for name in ("status.json", "manifest.json", "cache_audit.json", "missing_pairs.json"):
                if (blocks / name).is_file():
                    shutil.copyfile(blocks / name, diagnostic / name)
        code = 2
    result["elapsed_seconds"] = time.monotonic() - started
    json_write(attempt / "workflow_status.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("handoff", "qwen-cache", "scratch", "attempt"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.attempt.mkdir(parents=True, exist_ok=True)
    if not 0 < args.seconds <= 10800:
        parser.error("Use a CPU worker budget of at most three hours")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
