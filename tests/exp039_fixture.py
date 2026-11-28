"""EXP-039 fixture: an EXP-038 fixture run (mock GPU) becomes the sealed 'run 4' that the EXP-039 overlay carries."""

import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import exp038_fixture as fx38  # noqa: E402
from st3 import exp038 as science  # noqa: E402
from st3 import exp038_gpu as gpu  # noqa: E402
from st3 import exp039 as plan39  # noqa: E402
from st3.exp038_io import Clock, encode_members, restore_overlay  # noqa: E402

HEX64, HEX40 = "a" * 64, "b" * 40


def fake_e5_pins():
    return {"model": "Qwen/Qwen3-VL-Embedding-8B", "revision": "c" * 40, "license": "apache-2.0",
            "weights_sha256": {"model-00001-of-00004.safetensors": HEX64},
            "metadata_git_blobs": {"config.json": HEX40}}


CHECKPOINT = {"E3": ["model.language_model.embed_tokens.weight", "model.visual.patch_embed.proj.weight"],
              "E5": ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight",
                     "model.visual.patch_embed.proj.weight"]}
CONFIGS = {"E3": {"model_type": "qwen3_vl", "tie_word_embeddings": True, "text_config": {"tie_word_embeddings": True}},
           "E5": {"model_type": "qwen3_vl", "tie_word_embeddings": False, "text_config": {}}}


def git_blob(content):
    return hashlib.sha1(f"blob {len(content)}\0".encode() + content, usedforsecurity=False).hexdigest()


def audit_files():
    out = {}
    for arm in ("E3", "E5"):
        out[f"keys/{arm}_config.json"] = json.dumps(CONFIGS[arm]).encode()
        out[f"keys/{arm}_checkpoint_keys.json"] = json.dumps(CHECKPOINT[arm]).encode()
    return out


def fake_class_keys(config):
    """Stand-in for building Qwen3VLForConditionalGeneration on meta: the checkpoint names plus lm_head."""
    arm = "E3" if config.get("tie_word_embeddings") else "E5"
    return sorted([*CHECKPOINT[arm], "lm_head.weight"]), ("lm_head.weight",)


def build(tmp):
    import make_exp039_config
    import package_exp039
    import run_exp038

    base38 = fx38.build(tmp / "x38")
    ctx38 = fx38.context(base38, tmp / "r38")
    with mock.patch.object(science, "select", lambda gates, order, tolerance: "C2"):
        run_exp038.run_pipeline(ctx38, fx38.MockBackend(ctx38))
    results = ctx38.output / "exp038_results.zip"
    e3 = base38["config"]["arms"]["E3"]
    config = make_exp039_config.build(base38["config"], results, fake_e5_pins(),
                                      {k: e3[k] for k in ("revision", "weights_sha256", "metadata_git_blobs")})
    config["arms"]["E5"] |= {"width": 5, "batch": 3}
    keys = audit_files()
    for arm in ("E3", "E5"):
        config["arms"][arm]["metadata_git_blobs"] = config["arms"][arm]["metadata_git_blobs"] | {
            "config.json": git_blob(keys[f"keys/{arm}_config.json"])}
    config["arms"]["E6"]["batch"] = 3
    config["arms"]["E7"]["batch"] = 3
    members = package_exp039.overlay_members(base38["logical"], results, {"runtime/scripts/run_exp039.py": b"#\n"},
                                             config["references"]) | keys
    stored, encoded = encode_members(members)
    mounted = tmp / "mounted39"
    for name, content in stored.items():
        (mounted / name).parent.mkdir(parents=True, exist_ok=True)
        (mounted / name).write_bytes(content)
    manifest = {"source": "fixture39", "base": {"manifest_sha256": base38["base_sha"]}, "references": config["references"],
                "files": {n: hashlib.sha256(c).hexdigest() for n, c in sorted(stored.items())}, "encoded": encoded,
                "population": {}}
    content = (json.dumps(manifest, indent=2) + "\n").encode()
    (mounted / "exp039_manifest.json").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    restored = restore_overlay(mounted, digest, base38["base_sha"], tmp / "overlay39", manifest_name="exp039_manifest.json")
    return {"base": base38["base"], "base_sha": base38["base_sha"], "overlay": restored, "mounted": mounted,
            "overlay_sha": digest, "config": config, "track1": base38["track1"], "neighborhood": base38["neighborhood"],
            "members": members, "results38": results}


def rebuild(built, tmp, change):
    """Re-encode the overlay after ``change(members, config)`` and return a new built dict."""
    members, config = dict(built["members"]), copy.deepcopy(built["config"])
    change(members, config)
    stored, encoded = encode_members(members)
    mounted = tmp / "mounted_changed"
    for name, content in stored.items():
        (mounted / name).parent.mkdir(parents=True, exist_ok=True)
        (mounted / name).write_bytes(content)
    manifest = {"source": "fixture39", "base": {"manifest_sha256": built["base_sha"]}, "references": config["references"],
                "files": {n: hashlib.sha256(c).hexdigest() for n, c in sorted(stored.items())}, "encoded": encoded,
                "population": {}}
    content = (json.dumps(manifest, indent=2) + "\n").encode()
    (mounted / "exp039_manifest.json").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    restored = restore_overlay(mounted, digest, built["base_sha"], tmp / "overlay_changed",
                               manifest_name="exp039_manifest.json")
    return built | {"overlay": restored, "mounted": mounted, "overlay_sha": digest, "config": config, "members": members}


class Backend:
    def __init__(self, ctx, gpu_count=2, fail=None, nan_arms=(), rates=None):
        self.ctx, self.gpu_count = ctx, gpu_count
        self.fail, self.nan_arms = fail or {}, set(nan_arms)
        self.rates = rates or {}
        self.calls = []

    def runtime(self):
        expected = self.ctx.config["runtime"]
        return {"python": expected["python"], "gpu": "Mock T4", "gpu_count": self.gpu_count,
                "packages": {"torch": expected["torch"], "transformers": expected["transformers"],
                             "qwen-vl-utils": expected["qwen_vl_utils"]}}

    def fetch(self, arm):
        self.calls.append(("fetch", arm))
        if ("fetch", arm) in self.fail:
            raise self.fail[("fetch", arm)]
        return {"model": self.ctx.config["arms"][arm]["model"]}

    def benchmark(self, arm, items):
        self.calls.append(("benchmark", arm))
        if ("benchmark", arm) in self.fail:
            raise self.fail[("benchmark", arm)]
        return {"items_per_second": self.rates.get(arm, 1e4), "deterministic_within_1e-3": True, "load_seconds": 1.}

    def extract(self, arm, items, folder, identity, deadline, cap_deadline=None):
        import run_exp039

        self.calls.append(("extract", arm))
        if arm in self.fail:
            raise self.fail[arm]
        spec = run_exp039.spec(self.ctx, arm)
        embed = fx38.MockBackend.embed_for(arm, spec["width"])
        if arm in self.nan_arms:
            clean = embed

            def embed(images, instructions):
                values = clean(images, instructions)
                values[0, 0] = np.nan
                return values
        gpu.extract(arm, spec, embed, [tuple(i) for i in items], folder, identity, self.ctx.config["chunk_items"],
                    deadline, cap_deadline=cap_deadline)


def context(built, tmp, hours=10.):
    import run_exp039

    return run_exp039.Context(base=built["base"], overlay=built["overlay"], root=tmp / "working/exp039_run",
                              output=tmp / "working", base_sha=built["base_sha"], overlay_sha=built["overlay_sha"],
                              clock=Clock(time.time(), hours, 60), config=copy.deepcopy(built["config"]),
                              track1=copy.deepcopy(built["track1"]), neighborhood=copy.deepcopy(built["neighborhood"]),
                              echo=False)


__all__ = ["build", "rebuild", "Backend", "context", "plan39"]
