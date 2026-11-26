"""Resume the frozen Qwen extraction and hand verified records to the existing CV."""

import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd
from filelock import FileLock

from . import competition_semantic_recovery as recovery_source
from .baseline import file_sha256
from .cli import ROOT, json_output
from .competition_challenger import check_hash
from .competition_semantic_generation import CONFIG, TRACK1, verified_assets, verified_rgb
from .competition_semantic_recovery import RECOVERY, batch_profile, load_model, parse_compact
from .competition_track1 import load_protocol
from .e50_heads import read_json
from .e50_inference import safe_image_path

EVALUATION = ROOT / "configs/competition_semantic_evaluation.json"
IMAGE_KEYS = ("painting", "image", "image_sha256")


def json_digest(item):
    return hashlib.sha256(json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json_output(value))
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def validate_record(record, image, fields, token_cap):
    if any(record.get(key) != image[key] for key in IMAGE_KEYS):
        raise ValueError(f"Cached row has a different artwork identity: {image['painting']}")
    if not isinstance(record.get("raw"), str) or not 0 < record.get("output_tokens", 0) <= token_cap:
        raise ValueError(f"Cached row has invalid generation metadata: {image['painting']}")
    try:
        parsed = parse_compact(record["raw"], fields)
    except ValueError as error:
        if record.get("parse_error") != str(error) or "description" in record:
            raise ValueError("Invalid output receipt does not reproduce its parser failure") from error
        # An empty document produces the predictor's existing all-zero feature vector.
        return {}
    if parsed != record.get("description") or "parse_error" in record:
        raise ValueError("Cached semantic text differs from the strict parser result")
    return parsed


def frozen_work(args):
    rule = read_json(EVALUATION)
    check_hash(CONFIG, rule["base_config_sha256"])
    check_hash(RECOVERY, rule["recovery_config_sha256"])
    check_hash(Path(recovery_source.__file__), rule["recovery_source_sha256"])
    check_hash(args.cached, rule["batch_six_sha256"])
    cached = read_json(args.cached)
    if (cached["status"] != "MEASURED" or not cached["repeat_exact"]
            or cached["batch_size"] != rule["batch_size"] or len(cached["batches"]) != 2):
        raise ValueError("The verified compact batch-six cache is required")
    if rule["invalid_description"] != "retain_raw_and_use_zero_semantic_vector":
        raise ValueError("Unexpected handling of invalid semantic output")
    base, recipe, track1 = read_json(CONFIG), read_json(RECOVERY), read_json(TRACK1)
    protocol = load_protocol(args.inputs, track1).sort_values("painting")
    if (len(protocol) != track1["official_training"]["paintings"]
            or not protocol.loc[protocol.role.eq("development")].groupby("fold").size().eq(
                track1["p2"]["paintings_per_fold"]).all()):
        raise ValueError("Training image counts do not match the pinned official protocol")
    images = {row["painting"]: row for row in protocol[list(IMAGE_KEYS)].to_dict("records")}
    seed = cached["batches"][0]["records"]
    if len(seed) != rule["batch_size"] or seed != cached["batches"][1]["records"]:
        raise ValueError("The measured sample does not have an exact batch replay")
    for record in seed:
        if not validate_record(record, images[record["painting"]], base["fields"], recipe["max_new_tokens"]):
            raise ValueError("Cached grounding sample contains a malformed description")
    seeded = {item["painting"] for item in seed}
    if len(seeded) != len(seed) or not seeded.issubset(set(protocol.loc[protocol.role.eq("development"), "painting"])):
        raise ValueError("Cached sample must contain distinct paintings from P2")
    remaining = [row for name, row in images.items() if name not in seeded]
    width = rule["batch_size"]
    batches = [[images[row["painting"]] for row in seed], *[remaining[i:i + width] for i in range(0, len(remaining), width)]]
    identity = {"evaluation": file_sha256(EVALUATION), "recipe": file_sha256(RECOVERY), "base": file_sha256(CONFIG),
                "generator": file_sha256(Path(recovery_source.__file__)), "track1": file_sha256(TRACK1),
                "extraction_source": file_sha256(Path(__file__)),
                "predictor_source": file_sha256(Path(__file__).with_name("competition_semantic_vlm.py")),
                "checkpoint_revision": base["revision"], "cached": rule["batch_six_sha256"],
                "inventory": track1["p2_input_sha256"]["training/protocol.csv"], "schedule": json_digest(batches)}
    return rule, base, recipe, track1, images, seed, batches, identity


def batch_file(root, index):
    return root / "batches" / f"{index:04}.json"


def verify_batch(path, rows, identity, base, recipe):
    value = read_json(path)
    result = value["result"]
    if (value["identity"] != identity or value["result_sha256"] != json_digest(result)
            or len(result["records"]) != len(rows)):
        raise ValueError(f"Corrupt or incompatible batch {path}")
    return {image["painting"]: validate_record(record, image, base["fields"], recipe["max_new_tokens"])
            for image, record in zip(rows, result["records"], strict=True)}


def save_batch(path, identity, result):
    write_atomic(path, {"identity": identity, "result_sha256": json_digest(result), "result": result})


def prepare(args, plan):
    _, base, recipe, _, images, seed, batches, identity = plan
    for image in images.values():
        safe_image_path(args.data_dir, image["image"])
    schedule = args.run_dir / "schedule.json"
    if schedule.exists() and read_json(schedule) != identity:
        raise ValueError("Existing run has different source, settings, or artwork schedule")
    write_atomic(schedule, identity)
    path = batch_file(args.run_dir, 0)
    if not path.exists():
        for row in seed:
            verified_rgb(safe_image_path(args.data_dir, row["image"]), row["image_sha256"])
        save_batch(path, identity, {"records": seed, "reused": True, "wall_seconds": 0.0})
    available = []
    for i, rows in enumerate(batches):
        p = batch_file(args.run_dir, i)
        if p.exists():
            verify_batch(p, rows, identity, base, recipe)
            available.append(i)
    actual = set((args.run_dir / "batches").glob("*.json"))
    if actual != {batch_file(args.run_dir, i) for i in available}:
        raise ValueError("Unexpected batch files in run directory")
    if [0, *sorted(set(available) - {0})] != list(range(len(available))):
        raise ValueError("Extraction batches are missing before a completed batch")
    print(f"Verified {sum(len(batches[i]) for i in available)}/{len(images)} cached training artworks", flush=True)
    return len(available)


def worker(args, deadline):
    """An independent deadline also stops the worker if its supervisor is interrupted."""
    began = time.perf_counter()
    stop_event = threading.Event()

    def watchdog():
        if not stop_event.wait(max(0, deadline - time.time())):
            os._exit(124)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        with FileLock(str(args.run_dir / "worker.lock"), timeout=0):
            plan = frozen_work(args)
            rule, base, recipe, _, images, _, batches, identity = plan
            first = prepare(args, plan)
            if first == len(batches):
                return
            assets = verified_assets(args.checkpoint, base)
            import torch

            if not torch.cuda.is_available() or rule["gpu_name_fragment"] not in torch.cuda.get_device_name(0):
                raise ValueError("The owner-authorized RTX 4060 Laptop GPU must be CUDA device 0")
            torch, model, processor, hardware = load_model(args.checkpoint, base)
            model.set_attn_implementation(recipe["attention"])
            attempt = args.run_dir / "workers" / f"{time.time_ns()}.json"
            write_atomic(attempt, {"hardware": hardware, "asset_sha256": assets, "first_batch": first})
            for i in range(first, len(batches)):
                if time.time() >= deadline:
                    raise TimeoutError("Authorized cumulative GPU budget exhausted")
                completed = batch_profile(torch, model, processor, args.data_dir, batches[i], base, recipe, trace=False)
                save_batch(batch_file(args.run_dir, i), identity, completed)
                count = min(sum(len(group) for group in batches[:i + 1]), len(images))
                print(f"Extraction {count}/{len(images)} artworks, batch {i}, {completed['wall_seconds']:.1f} s, "
                      f"budget remaining {max(0, deadline - time.time()) / 3600:.2f} h", flush=True)
            print(f"GPU worker completed in {time.perf_counter() - began:.2f} seconds", flush=True)
    finally:
        stop_event.set()


def charge(ledger, now):
    active = ledger.get("active_since")
    return min(ledger["limit_seconds"], ledger["charged_seconds"] + (max(0, now - active) if active is not None else 0))


def supervised_worker(args, command, plan):
    rule, *_, identity = plan
    path = args.run_dir / "gpu_budget.json"
    with FileLock(str(args.run_dir / "worker.lock"), timeout=0):
        if path.exists():
            ledger = read_json(path)
            if ledger["limit_seconds"] != rule["max_gpu_wall_seconds"] or ledger["identity"] != identity:
                raise ValueError("Run budget receipt is bound to a different plan")
        else:
            ledger = {"limit_seconds": rule["max_gpu_wall_seconds"], "identity": identity, "charged_seconds": 0.0}
        start = time.time()
        ledger.update(charged_seconds=charge(ledger, start), active_since=None)
        remaining = ledger["limit_seconds"] - ledger["charged_seconds"]
        if remaining <= 0:
            write_atomic(path, ledger)
            raise TimeoutError("The 3.5-hour cumulative GPU job budget is exhausted")
        ledger.update(active_since=start, last_exit=None)
        write_atomic(path, ledger)
    process = None
    try:
        deadline = start + remaining
        process = subprocess.Popen([*command, "--worker-deadline", str(deadline)])
        code = process.wait(timeout=max(0, deadline - time.time()))
        if code:
            raise RuntimeError(f"GPU worker exited with code {code}; completed batches are retained")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        ledger.update(charged_seconds=charge(ledger, time.time()), active_since=None,
                      last_exit=process.returncode if process is not None else None)
        write_atomic(path, ledger)


def sealed_recovery(args, _config=None, _track1=None):
    _, base, recipe, track1, images, _, batches, identity = frozen_work(args)
    all_descriptions, hashes = {}, {}
    invalid = []
    for i, rows in enumerate(batches):
        path = batch_file(args.run_dir, i)
        if not path.is_file():
            raise FileNotFoundError(f"Training semantics are incomplete at batch {i}")
        parsed = verify_batch(path, rows, identity, base, recipe)
        hashes[path.name] = file_sha256(path)
        all_descriptions.update(parsed)
        invalid.extend(name for name, description in parsed.items() if not description)
    if set(all_descriptions) != set(images):
        raise ValueError("Sealed semantic cache does not cover the full training population")
    path = args.run_dir / "extraction.json"
    receipt = {"identity": identity, "batch_sha256": hashes, "training_artworks": len(images),
               "schema_valid": len(images) - len(invalid), "schema_invalid": invalid}
    if path.exists() and read_json(path) != receipt:
        raise ValueError("Training semantic extraction seal changed")
    if not path.exists():
        write_atomic(path, receipt)
    protocol = load_protocol(args.inputs, track1)
    p2 = protocol.loc[protocol.role.eq("development"), "painting"]
    return pd.Series({name: all_descriptions[name] for name in p2}, dtype=object), file_sha256(path)
