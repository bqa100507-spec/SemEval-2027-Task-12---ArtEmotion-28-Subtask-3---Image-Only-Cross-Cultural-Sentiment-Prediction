"""Small synthetic EXP-037 base and EXP-038 overlay, plus a deterministic GPU backend mock."""

import copy
import hashlib
import io
import json
import sys
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from st3 import data  # noqa: E402
from st3 import exp038 as science  # noqa: E402
from st3 import exp038_gpu as gpu  # noqa: E402
from st3.competition_challenger import frame_sha  # noqa: E402
from st3.competition_stack import ENCODERS  # noqa: E402
from st3.exp036_io import read  # noqa: E402
from st3.exp038_io import Clock, base_features, painting_images, write_submission  # noqa: E402
from st3.metrics import score  # noqa: E402

LANGUAGES = ["Arabic", "English", "Hausa", "Korean", "Thai"]
WIDTHS = {"clip": 6, "siglip2": 7, "clip_large": 8}
ARM_WIDTHS = {"E1": 5, "E2": 4, "E3": 6}
POS, NEG = sorted(data.POSITIVE), sorted(data.NEGATIVE)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def configs():
    config = read(ROOT / "configs/exp038.json")
    track1 = read(ROOT / "configs/competition_track1.json")
    neighborhood = read(ROOT / "configs/competition_neighborhood_residual.json")
    config = copy.deepcopy(config)
    for arm, width in ARM_WIDTHS.items():
        config["arms"][arm]["width"] = width
        config["arms"][arm]["batch"] = 3
    config["arms"]["E4"]["batch"] = 3
    config["chunk_items"] = 7
    track1 = copy.deepcopy(track1) | {"feature_widths": dict(WIDTHS)}
    return config, track1, neighborhood


def jpeg(color):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=tuple(int(c) for c in color)).save(buffer, format="JPEG")
    return buffer.getvalue()


def votes(rng, latent, n):
    k = rng.binomial(n, 1 / (1 + np.exp(-latent)))
    return k


def build(tmp, seed=0):
    rng = np.random.default_rng(seed)
    config, track1, neighborhood = configs()
    base, overlay = tmp / "base", tmp / "overlay"
    for folder in (base, overlay):
        folder.mkdir(parents=True)
    p2 = [f"p{i:02d}" for i in range(50)]
    holdout = [f"h{i}" for i in range(6)]
    dev_names = [f"d{i}" for i in range(8)]
    files, images = {}, {}

    def put(root, name, content, registry):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        registry[name] = digest(content)

    logical = {}
    for name in [*p2, *holdout, *dev_names]:
        content = jpeg(rng.integers(0, 255, 3))
        physical = f"artwork/{digest(content)}.jpg"
        logical[name] = f"Style/{name} 'x'.jpg"
        put(base, physical, content, files)
        images[logical[name]] = physical
    rows = []
    for name in [*p2, *holdout]:
        chosen = rng.choice(LANGUAGES, size=rng.integers(3, 6), replace=False)
        base_level = rng.normal()
        for language in chosen:
            n = int(rng.integers(5, 8))
            k = votes(rng, base_level + rng.normal(scale=.8), n)
            for v in range(n):
                emotion = POS[v % 4] if v < k else NEG[v % 4]
                rows.append((name, language, logical[name], "train", emotion, "caption"))
        rows.append((name, chosen[0], logical[name], "train", "other", "caption"))
    annotations = pd.DataFrame(rows, columns=[*data.ANNOTATION_COLUMNS, "caption"])
    put(base, "training/annotations.csv", annotations.to_csv(index=False).encode(), files)
    dev_rows, gold_rows = [], []
    for name in dev_names:
        for language in rng.choice(LANGUAGES, size=rng.integers(3, 6), replace=False):
            n = int(rng.integers(5, 8))
            dev_rows.append((name, language, logical[name]))
            gold_rows.append((name, language, int(votes(rng, rng.normal(), n)), n))
    dev = pd.DataFrame(dev_rows, columns=data.INPUT_COLUMNS)
    put(base, data.DEV_INPUT, dev.to_csv(index=False, lineterminator="\n").encode(), files)
    put(base, "scoring/dev_gold.csv", pd.DataFrame(gold_rows, columns=[*data.KEYS, "k", "n"]).to_csv(
        index=False, lineterminator="\n").encode(), files)
    buffer = io.BytesIO()
    np.savez(buffer, values=np.zeros((1, 2), np.float32), mask=np.zeros(1, bool))
    put(base, "training/targets.npz", buffer.getvalue(), files)
    manifest = (json.dumps({"files": files, "images": images, "source": "fixture",
                            "population": {"train_paintings": 56}}, indent=2) + "\n").encode()
    (base / "exp037_manifest.json").write_bytes(manifest)
    base_sha = digest(manifest)

    train = data.training_pairs(annotations[data.ANNOTATION_COLUMNS]).sort_values(data.KEYS, ignore_index=True)
    dev = data.read_dev_input(base)
    overlay_files = {}
    image_sha = {name: images[logical[name]].split("/")[1][:-4] for name in logical}
    protocol = pd.DataFrame([(n, logical[n], image_sha[n], "development", 1 + i % 5) for i, n in enumerate(p2)] +
                            [(n, logical[n], image_sha[n], "holdout", 0) for n in holdout],
                            columns=["painting", "image", "image_sha256", "role", "fold"])
    put(overlay, "p2/protocol.csv", protocol.to_csv(index=False).encode(), overlay_files)
    gold = train.loc[train.painting.isin(p2)].copy()
    gold["fold"] = gold.painting.map(protocol.set_index("painting").fold)
    gold["split"] = "train"
    put(overlay, "p2/pairs.csv", gold.to_csv(index=False).encode(), overlay_files)
    shas = sorted(image_sha.values())
    arrays = {"image_sha256": np.asarray(shas)}
    for name, width in WIDTHS.items():
        x = rng.standard_normal((len(shas), width)).astype(np.float32)
        arrays[name] = x / np.linalg.norm(x, axis=1, keepdims=True)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    put(overlay, "features/base_features.npz", buffer.getvalue(), overlay_files)

    refs = {}
    paintings = painting_images(base, json.loads(manifest), train, dev)
    features = base_features(overlay, paintings, WIDTHS)
    gold = gold[data.KEYS + ["fold", "k", "n"]].sort_values(data.KEYS, ignore_index=True)
    inner = {}
    for fold in range(1, 6):
        _, details = science.crossfit(gold.loc[gold.fold.ne(fold)], features, None, (), track1, fold, {})
        inner[str(fold)] = details["r0_components_sha256"]
    frame, parts, r0, _, _ = science.evaluate_p2(gold, features, None, (), config, track1, neighborhood, inner,
                                                 log=lambda *_: None)
    oof = parts[data.KEYS + list(ENCODERS) + ["fixed", "fold"]]
    put(overlay, "reference/track1_oof.csv", oof.to_csv(index=False, lineterminator="\n", float_format="%.6f").encode(),
        overlay_files)
    text = science.csv_text(r0)
    put(overlay, "reference/corrected_oof.csv", text.encode(), overlay_files)
    t1m, nm = {}, {}
    for scope, subset in science.scopes(gold, config["test_like_excluded"]):
        for column, source, target in (("fixed", parts, t1m), ("corrected", r0, nm)):
            values = subset[data.KEYS].merge(source[data.KEYS + [column]], on=data.KEYS)[column].to_numpy()
            result = score(subset[data.KEYS + ["k", "n"]], subset[data.KEYS].assign(p_positive=values))
            pooled = {k: result[k] for k in ("cultural_contrast", "log_loss")}
            if column == "fixed":
                target[scope] = {"systems": {"fixed": {"pooled": pooled}}}
            else:
                target[scope] = {"pooled": {"corrected": pooled}}
    put(overlay, "reference/track1_metrics.json", json.dumps(t1m).encode(), overlay_files)
    put(overlay, "reference/corrected_metrics.json", json.dumps(nm).encode(), overlay_files)

    heads, banks, _, visual = science.deployment_fit(train, features, None, (), track1, neighborhood, {})
    dev_frame, _ = science.development_predictions(heads, banks, visual, dev[data.KEYS], features, None, {},
                                                   neighborhood)
    scratch = tmp / "scratch"
    fixed = write_submission(science.system_frame(dev_frame, "R0_fixed"), dev, scratch, "fixed")
    put(overlay, "reference/fixed_ridge_submission.zip", fixed.read_bytes(), overlay_files)
    corrected = science.system_frame(dev_frame, "R0_corrected")
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as z:
        z.writestr("predictions.csv", science.csv_text(corrected))
    put(overlay, "reference/corrected_ridge_submission.zip", buffer.getvalue(), overlay_files)
    bank = banks["R0"]
    buffer = io.BytesIO()
    np.savez_compressed(buffer, paintings=bank.visual.index.to_numpy(dtype=str),
                        languages=bank.votes.columns.to_numpy(dtype=str), image_sha256=np.asarray(["x"]),
                        visual=bank.visual.to_numpy(), numerator=bank.numerator.to_numpy(), votes=bank.votes.to_numpy())
    put(overlay, "reference/corrected_bundle/bank.npz", buffer.getvalue(), overlay_files)
    put(overlay, "reference/corrected_bundle/manifest.json",
        json.dumps({"bank_sha256": overlay_files["reference/corrected_bundle/bank.npz"]}).encode(), overlay_files)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{f"{e}:{f}": np.asarray(getattr(heads[e], f)) for e in ENCODERS
                                   for f in ("intercept", "language_intercepts", "shared", "deviations", "languages")})
    put(overlay, "reference/fixed_bundle/heads.npz", buffer.getvalue(), overlay_files)
    put(overlay, "reference/fixed_bundle/manifest.json", b"{}", overlay_files)
    cs = dev[data.KEYS].assign(p_positive=np.round(rng.uniform(.2, .8, len(dev)), 6))
    c6 = dev[data.KEYS].assign(p_positive=np.round(rng.uniform(.2, .8, len(dev)), 6))
    put(overlay, "reference/exp036_standalone_submission.zip",
        write_submission(cs, dev, scratch, "cs").read_bytes(), overlay_files)
    composed = science.blends({"R0": corrected}, {"original": cs}, c6, dev, config)
    put(overlay, "reference/exp036_ensemble_submission.zip",
        write_submission(composed["B1_R0_original"], dev, scratch, "b1").read_bytes(), overlay_files)
    put(overlay, "reference/exp037_best_submission.zip",
        write_submission(composed["B2_R0_original"], dev, scratch, "b2").read_bytes(), overlay_files)
    put(overlay, "reference/exp037_c6_standalone/predictions.csv",
        c6.to_csv(index=False, lineterminator="\n", float_format="%.6f").encode(), overlay_files)
    refs = config["references"] | {
        "base_manifest_sha256": base_sha, "p2_protocol_sha256": overlay_files["p2/protocol.csv"],
        "p2_pairs_sha256": overlay_files["p2/pairs.csv"], "track1_inner_oof_sha256": inner,
        "corrected_oof_sha256": digest(text.encode()), "training_pair_sha256": frame_sha(train[data.KEYS + ["image", "k", "n"]]),
        "corrected_bank_sha256": overlay_files["reference/corrected_bundle/bank.npz"],
        "exp037_original_blend_zip_sha256": overlay_files["reference/exp037_best_submission.zip"],
        "exp036_standalone_zip_sha256": overlay_files["reference/exp036_standalone_submission.zip"],
        "exp037_c6_standalone_csv_sha256": overlay_files["reference/exp037_c6_standalone/predictions.csv"],
        "caption_targets_sha256": files["training/targets.npz"]}
    config["references"] = refs
    om = (json.dumps({"source": "fixture-source", "base": {"manifest_sha256": base_sha}, "files": overlay_files,
                      "references": refs, "population": {}}, indent=2) + "\n").encode()
    (overlay / "exp038_manifest.json").write_bytes(om)
    import shutil

    shutil.rmtree(scratch)
    return {"base": base, "overlay": overlay, "base_sha": base_sha, "overlay_sha": digest(om), "config": config,
            "track1": track1, "neighborhood": neighborhood}


class MockBackend:
    """Deterministic stand-in for every GPU child: no models, no downloads."""

    def __init__(self, ctx, rates=None, fail=None, nan_arms=()):
        self.ctx = ctx
        self.nan_arms = set(nan_arms)
        self.rates = rates or {"E1": 1e4, "E2": 1e4, "E3": 1e4, "E4": 1e4}
        self.fail = fail or {}
        self.calls = []

    def runtime(self):
        expected = self.ctx.config["runtime"]
        return {"python": expected["python"], "gpu": "Mock T4",
                "packages": {"torch": expected["torch"], "transformers": expected["transformers"],
                             "qwen-vl-utils": expected["qwen_vl_utils"], "numpy": np.__version__}}

    def fetch_models(self):
        self.calls.append(("fetch",))
        return {"E1": "mock", "E2": "mock", "E3": "mock"}

    def benchmark(self, arm, items):
        self.calls.append(("benchmark", arm))
        return {"arm": arm, "items": len(items), "items_per_second": self.rates[arm],
                "deterministic_within_1e-3": True, "bitwise_deterministic": True}

    @staticmethod
    def embed_for(arm, width):
        def embed(images, instructions):
            rows = []
            for i, image in enumerate(images):
                text = instructions[i] if instructions else ""
                key = hashlib.sha256(np.asarray(image).tobytes() + arm.encode() + text.encode()).digest()
                x = np.random.default_rng(int.from_bytes(key[:8], "little")).standard_normal(width)
                rows.append(x / np.linalg.norm(x))
            return np.asarray(rows, dtype=np.float32)
        return embed

    def extract(self, arm, items, folder, identity, deadline, cap_deadline=None):
        self.calls.append(("extract", arm))
        if arm in self.fail:
            raise self.fail[arm]
        spec = self.ctx.config["arms"][arm]
        if arm == "E4":
            spec = self.ctx.config["arms"]["E3"] | spec
        embed = self.embed_for(arm, spec["width"])
        if arm in self.nan_arms:
            clean = embed

            def embed(images, instructions):
                values = clean(images, instructions)
                values[0, 0] = np.nan
                return values
        gpu.extract(arm, spec, embed, [tuple(i) for i in items], folder, identity,
                    self.ctx.config["chunk_items"], deadline, cap_deadline=cap_deadline)

    def cscva(self, seed, folder, deadline, preflight=False):
        self.calls.append(("cscva", seed, preflight))
        if preflight:
            return {"paintings_per_second": 10., "forecast_fit_seconds": 60., "microbatch": 4}
        if seed in self.fail:
            raise self.fail[seed]
        dev = data.read_dev_input(self.ctx.base)
        rng = np.random.default_rng(seed)
        folder.mkdir(parents=True, exist_ok=True)
        dev[data.KEYS].assign(p_positive=np.round(rng.uniform(.2, .8, len(dev)), 6)).to_csv(
            folder / "predictions.csv", index=False, lineterminator="\n", float_format="%.6f")
        (folder / "inference").mkdir(exist_ok=True)
        (folder / "inference/weights.pt").write_bytes(b"mock")
        (folder / "inference/bundle.json").write_text(json.dumps({"epoch": 5, "seed": seed}))
        (folder / "train.json").write_text(json.dumps({"seed": seed, "freeze_epoch": 5}))
        return {}


def context(fx, tmp, hours=10., **kw):
    from run_exp038 import Context

    clock = Clock(time.time(), hours, kw.pop("reserve", 60))
    return Context(base=fx["base"], overlay=kw.get("overlay", fx["overlay"]), root=tmp / "working/exp038_run",
                   output=tmp / "working", base_sha=fx["base_sha"], overlay_sha=kw.get("overlay_sha", fx["overlay_sha"]),
                   clock=clock, config=copy.deepcopy(kw.get("config", fx["config"])),
                   track1=copy.deepcopy(fx["track1"]), neighborhood=copy.deepcopy(fx["neighborhood"]), echo=False)
