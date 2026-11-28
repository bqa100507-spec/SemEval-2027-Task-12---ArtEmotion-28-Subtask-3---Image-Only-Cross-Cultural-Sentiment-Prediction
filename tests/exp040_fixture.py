"""EXP-040 fixture: the EXP-039 fixture overlay becomes the carried input of a synthetic EXP-040 overlay.

The GPU child is mocked: it reads the exact request the worker writes and returns what the real child would
(step-0 receipts, smoke/benchmark throughput, fold histories, adapted embeddings and held-out probabilities).
"""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import exp039_fixture as fx39  # noqa: E402
from st3 import exp040 as plan  # noqa: E402
from st3.competition_ridge import Head  # noqa: E402
from st3.exp036_io import write  # noqa: E402
from st3.exp038_io import Clock, encode_members, restore_overlay  # noqa: E402


def zip_dir(folder, archive, prefix):
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as z:
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file():
                z.write(path, prefix + path.relative_to(folder).as_posix())
    return archive


def build(tmp):
    import make_exp040_config
    import package_exp040

    built39 = fx39.build(tmp / "x39")
    overlay39 = zip_dir(built39["mounted"], tmp / "upload39/exp039_inputs.zip", "exp039_inputs/")
    handoff39 = {"archive_sha256": hashlib.sha256(overlay39.read_bytes()).hexdigest(),
                 "manifest_sha256": built39["overlay_sha"]}
    config = make_exp040_config.build(built39["config"], overlay39, handoff39)
    config["smoke"] |= {"optimizer_steps": 3, "benchmark_steps": 2}
    members = package_exp040.carried_members(built39["overlay"]) | {"runtime/scripts/run_exp040.py": b"#\n"}
    stored, encoded = encode_members(members)
    mounted = tmp / "mounted40"
    for name, content in stored.items():
        (mounted / name).parent.mkdir(parents=True, exist_ok=True)
        (mounted / name).write_bytes(content)
    manifest = {"source": "fixture40", "base": {"manifest_sha256": built39["base_sha"]},
                "references": config["references"],
                "files": {n: hashlib.sha256(c).hexdigest() for n, c in sorted(stored.items())}, "encoded": encoded,
                "population": {}}
    content = (json.dumps(manifest, indent=2) + "\n").encode()
    (mounted / "exp040_manifest.json").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    restored = restore_overlay(mounted, digest, built39["base_sha"], tmp / "overlay40", manifest_name="exp040_manifest.json")
    return built39 | {"overlay": restored, "mounted": mounted, "overlay_sha": digest, "config": config,
                      "members": members, "overlay39_zip": overlay39, "handoff39": handoff39}


def context(built, tmp, hours=10.):
    import run_exp040

    return run_exp040.Context(base=built["base"], overlay=built["overlay"], root=tmp / "working/exp040_run",
                              output=tmp / "working", base_sha=built["base_sha"], overlay_sha=built["overlay_sha"],
                              clock=Clock(time.time(), hours, 60), config=copy.deepcopy(built["config"]),
                              track1=copy.deepcopy(built["track1"]), neighborhood=copy.deepcopy(built["neighborhood"]),
                              echo=False)


def load_head(path):
    fields = dict(np.load(path, allow_pickle=False))
    return Head(float(fields["intercept"]), fields["language_intercepts"], fields["shared"], fields["deviations"],
                tuple(fields["languages"].tolist()))


class Backend:
    """Mock E3L child. ``epochs`` maps fold -> selected epoch (default 2); ``fail`` maps a mode/fold to an error."""

    def __init__(self, ctx, epochs=None, fail=None, rate=50., step_seconds=.01, gpu_count=2, cache_difference=0.,
                 entry_bytes=4_000_000):
        import threading

        self.ctx, self.epochs, self.fail = ctx, epochs or {}, fail or {}
        self.rate, self.step_seconds, self.gpu_count = rate, step_seconds, gpu_count
        self.cache_difference, self.entry_bytes = cache_difference, entry_bytes
        self.calls, self.requests, self.placements, self.cancels = [], [], [], 0
        self.lock = threading.Lock()

    def gpus(self):
        return list(range(self.gpu_count))

    def cancel(self):
        self.cancels += 1

    def runtime(self):
        expected = self.ctx.config["runtime"]
        return {"python": expected["python"], "gpu": "Mock T4", "gpu_count": self.gpu_count,
                "packages": {"torch": expected["torch"], "transformers": expected["transformers"],
                             "qwen-vl-utils": expected["qwen_vl_utils"]}}

    def fetch(self, arm):
        self.calls.append(("fetch", arm))
        return {"model": self.ctx.config["arms"][arm]["model"]}

    def sealed(self):
        import run_exp040

        return run_exp040.base_features(self.ctx)["E3"]

    def adapted(self, paintings, scale=.05, seed=0):
        e3 = self.sealed().loc[sorted(paintings)]
        rng = np.random.default_rng(seed)
        values = e3.to_numpy() + rng.normal(scale=scale, size=e3.shape)
        return pd.DataFrame(values / np.linalg.norm(values, axis=1, keepdims=True), index=e3.index)

    def e3l(self, request_path, deadline, gpu=0):
        body = json.loads(Path(request_path).read_text(encoding="utf-8"))
        mode = body["mode"]
        fold = body["identity"].get("fold")
        with self.lock:
            self.requests.append(body)
            self.calls.append(("e3l", mode, fold))
            self.placements.append((mode, fold, gpu))
        error = self.fail.get((mode, fold)) or self.fail.get(mode)
        if error is not None:
            raise error
        folder = Path(body["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        head = load_head(body["head"])
        pairs = pd.read_csv(body["pairs"])
        if mode == "step0":
            x = self.sealed().loc[sorted(body["held"])]
            keys = pairs[pairs.painting.isin(set(body["held"]))][["painting", "language"]].sort_values(
                ["painting", "language"], ignore_index=True)
            x.to_csv(folder / "step0_embeddings.csv", float_format="%.17g")
            keys.assign(p_positive=plan.ridge_probabilities(head, x, keys)).to_csv(
                folder / "step0_probabilities.csv", index=False, float_format="%.17g")
            receipt = {"paintings": len(x), "lora_right_zero": True, "placement": ["cuda:0"]}
            if (body.get("vision_cache") or {}).get("check"):
                d = self.cache_difference
                passes = {n: {"paintings_per_second": self.rate, "max_abs_probability_difference": d,
                              "max_abs_embedding_difference": d, "bitwise": d == 0} for n in ("fill", "hit")}
                receipt["vision_cache"] = {"passes": passes, "max_abs_probability_difference": d,
                                           "tolerance": body["vision_cache"]["tolerance"],
                                           "passed": d <= body["vision_cache"]["tolerance"],
                                           "cache": {"entries": len(x), "bytes_per_entry": self.entry_bytes,
                                                     "max_entry_bytes": self.entry_bytes}}
            write(folder / "step0.json", receipt)
            return
        if mode == "smoke":
            write(folder / "smoke.json", {"mode": "SMOKE_ONLY", "load_seconds": 1., "optimizer_steps": body["smoke_steps"],
                                          "seconds_per_optimizer_step": self.step_seconds,
                                          "warm_seconds_per_optimizer_step": self.step_seconds,
                                          "inner_paintings_per_second": self.rate, "finite_losses": True,
                                          "revisit_seconds_per_optimizer_step": self.step_seconds / 2,
                                          "revisit_inner_paintings_per_second": self.rate * 2,
                                          "gpu": {"visible_devices": str(gpu)},
                                          "peak_memory": {"allocated_bytes": 1}, "devices": {"hidden": "cuda:0"},
                                          "two_pass_training": {"bitwise": True}})
            return
        selected = body["epochs_to_run"] if mode == "full" else self.epochs.get(fold, 2)
        history = [{"epoch": e, "inner_test_like_contrast": (1. if e == selected else .5) if mode == "fold" else None}
                   for e in range(body["epochs_to_run"] + 1)]
        result = {"identity": body["identity"], "history": history, "selected_epoch": selected,
                  "train_paintings": len(body["train_paintings"])}
        if selected > 0:
            embeddings = self.adapted(body["embed"], seed=fold or 99)
            keys = pairs[pairs.painting.isin(set(body["held"]))][["painting", "language"]]
            probabilities = keys.assign(p_positive=plan.ridge_probabilities(head, embeddings, keys))
            np.savez(folder / "embeddings.npz", paintings=np.asarray(embeddings.index, dtype=str),
                     values=embeddings.to_numpy(dtype=np.float32))
            probabilities.to_csv(folder / "held_probabilities.csv", index=False, float_format="%.12g")
            (folder / "epochs").mkdir(exist_ok=True)
            (folder / "epochs" / f"epoch_{selected}.pt").write_bytes(b"mock weights")
        write(folder / "result.json", result)


__all__ = ["build", "context", "Backend", "load_head", "zip_dir"]
