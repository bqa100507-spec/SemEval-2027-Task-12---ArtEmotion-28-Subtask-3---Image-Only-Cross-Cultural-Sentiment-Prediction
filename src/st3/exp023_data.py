"""Handoff validation and historical reconstruction. Confirmation is a separate guarded read."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .caption_data import inner_partition, require_isolation
from .caption_student import concatenate_features
from .experiment_workflow import StopRun, sha256, fingerprint
from .submission import validate_predictions


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def load_banks(path):
    with np.load(path, allow_pickle=False) as archive:
        names = archive["names"].tolist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate image feature identities")
        banks = [dict(zip(names, archive[key])) for key in ("clip", "siglip", "large")]
    return banks, concatenate_features(banks)


def exact(left, right, columns=("painting", "language", "fold")):
    columns = list(columns)
    if left.duplicated(columns).any() or right.duplicated(columns).any():
        raise ValueError("Duplicate historical prediction keys")
    a, b = [frame.sort_values(columns).reset_index(drop=True) for frame in (left, right)]
    if not a[columns].equals(b[columns]) or not np.array_equal(a.p_positive.to_numpy(), b.p_positive.to_numpy()):
        raise StopRun("FAILED_HISTORICAL_PARITY", "Historical keys/folds/probabilities differ exactly")


def reconstruct_history(root, training):
    root = Path(root)
    verification = json.loads((root / "history/independent_verification.json").read_text())
    if (verification.get("status") != "PASS" or verification.get("checkpoints_replayed") != 45
            or verification.get("baseline_heads_refitted") != 15 or verification.get("pca_partitions_recomputed") != 20):
        raise StopRun("FAILED_HISTORICAL_PARITY", "Original EXP-017 independent verification is incomplete")
    old = read_csv(root / "history/exp017_predictions.csv")
    current = read_csv(root / "history/exp022_R0.csv")
    reference = old[old.model.eq("M0")]
    exact(reference, current)
    exact(reference, training[[*data.KEYS, "fold"]].merge(reference, on=[*data.KEYS, "fold"], validate="one_to_one"))
    if len(reference) != len(training):
        raise StopRun("FAILED_HISTORICAL_PARITY", "Historical training coverage differs")
    checks = 0
    # Historical seed CSVs contain rounded values, not raw probabilities. Reconstruct
    # the saved blends from authoritative rounded student means, never average the
    # rounded seeds and pretend this recovers the missing raw means.
    for fold in range(1, 6):
        fitting, held = training[training.fold.ne(fold)], training[training.fold.eq(fold)]
        inner, stopping = inner_partition(fitting, {"early_stop_seed": 17029, "early_stop_fraction": .1}, fold)
        partitions = json.loads((root / f"history/fold_{fold}_partition.json").read_text())
        for name, frame in (("fit", fitting), ("held", held), ("inner", inner), ("stopping", stopping)):
            if partitions[name] != sorted(frame.painting.unique()):
                raise StopRun("FAILED_HISTORICAL_PARITY", "Historical training/held-out partition identity differs")
        original_run = json.loads((root / f"history/fold_{fold}_run.json").read_text())
        if len(original_run["lr_fits"]) != 3 or not all(item["converged"] for item in original_run["lr_fits"]):
            raise StopRun("FAILED_HISTORICAL_PARITY", "Historical fold-isolated R0 components did not converge")
        inputs = reference[reference.fold.eq(fold)]
        raw = read_csv(root / f"history/seeds_fold_{fold}.csv")
        for model in ("M1", "M2"):
            rows = raw[raw.model.eq(model)]
            if set(rows.seed) != {2027, 2028, 2029}:
                raise StopRun("FAILED_HISTORICAL_PARITY", "Historical raw seed coverage incomplete")
            for seed in (2027, 2028, 2029):
                part = inputs[data.KEYS].merge(rows[rows.seed.eq(seed)], on=data.KEYS, validate="one_to_one")
                if (len(part) != len(inputs) or not np.isfinite(part.p_positive).all()
                        or not np.array_equal(part.p_positive, np.round(part.p_positive, 6))):
                    raise StopRun("FAILED_HISTORICAL_PARITY", "Historical raw probabilities invalid")
            student = inputs[[*data.KEYS, "fold"]].merge(
                old[old.fold.eq(fold) & old.model.eq(model)][[*data.KEYS, "p_positive"]],
                on=data.KEYS, validate="one_to_one")
            ensemble = student.assign(p_positive=np.round(.5 * student.p_positive.to_numpy() + .5 * inputs.p_positive.to_numpy(), 6))
            exact(ensemble, old[old.fold.eq(fold) & old.model.eq("B" + model[1:])])
            checks += 1
    return {"status": "PASS", "checks": checks, "reference_rows": len(reference),
            "reference_sha256": sha256(root / "history/exp022_R0.csv"),
            "boundary": "Exact stored-mean blend reconstruction; historical raw seeds were not persisted. Prior independent replay is separate evidence."}


def load_inputs(root, config):
    root = Path(root)
    manifest = json.loads((root / "input_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        # Sealed confirmation content is hashed as opaque bytes, never parsed here.
        if sha256(root / name) != digest:
            raise StopRun("BLOCKED_INPUT", f"Input checksum mismatch: {name}")
    train = read_csv(root / "training/pairs.csv")
    protocol = read_csv(root / "training/protocol.csv")
    locked = protocol[protocol.role.eq("holdout")]
    allowed = protocol[protocol.role.eq("development")]
    data.unique_pairs(train)
    if (len(train) != config["training_pairs"] or train.painting.nunique() != config["training_paintings"]
            or not train.split.eq("train").all() or not train.role.eq("development").all()
            or set(train.fold) != {1, 2, 3, 4, 5} or (train.n < 5).any()
            or (train.k < 0).any() or (train.k > train.n).any()):
        raise StopRun("BLOCKED_INPUT", "Frozen training population/counts differ")
    if len(locked) != config["confirmation_paintings"] or len(allowed) != config["training_paintings"]:
        raise StopRun("BLOCKED_INPUT", "Protocol painting counts differ")
    mapping = train[["painting", "image", "image_sha256", "fold"]].drop_duplicates().sort_values("painting").reset_index(drop=True)
    expected = allowed[mapping.columns].sort_values("painting").reset_index(drop=True)
    if not mapping.equals(expected) or mapping.groupby("fold").size().tolist() != [267] * 5:
        raise StopRun("BLOCKED_INPUT", "Original fold identity differs")
    require_isolation(train, locked)
    captions = read_csv(root / "training/captions.csv")
    if not set(captions.painting) <= set(train.painting) or not captions.split.eq("train").all():
        raise StopRun("BLOCKED_INPUT", "Caption coverage includes forbidden paintings")
    if len(captions) != 69719 or captions.caption_id.nunique() != 69153:
        raise StopRun("BLOCKED_INPUT", "Historical caption coverage differs")
    counts = captions.assign(positive=captions.emotion.isin(data.POSITIVE)).groupby(data.KEYS).agg(
        caption_n=("emotion", "size"), caption_k=("positive", "sum")).reset_index()
    joined = train.merge(counts, on=data.KEYS, validate="one_to_one")
    if (len(joined) != len(train) or not joined.n.eq(joined.caption_n).all() or not joined.k.eq(joined.caption_k).all()
            or not captions.emotion.isin(data.POSITIVE | data.NEGATIVE).all()):
        raise StopRun("BLOCKED_INPUT", "Repeated original caption votes do not reconstruct training targets")
    path = root / "text/features.npz"
    if sha256(path) != config["caption_cache_sha256"]:
        raise StopRun("BLOCKED_INPUT", "Pinned CLIP caption cache SHA differs")
    text_meta = json.loads((root / "text/run.json").read_text())
    if (text_meta["features_sha256"] != config["caption_cache_sha256"]
            or text_meta["identity"]["revision"] != config["pretrained"]["clip_large"]
            or set(text_meta["fit_paintings"]) != set(train.painting)):
        raise StopRun("BLOCKED_INPUT", "Caption source manifest differs")
    with np.load(path, allow_pickle=False) as a:
        names, values = a["keys"].tolist(), a["features"]
    if (len(set(names)) != len(names) or values.shape != (len(names), 768) or values.dtype != np.float32
            or not np.allclose(np.linalg.norm(values, axis=1), 1., rtol=0, atol=3e-6)):
        raise StopRun("BLOCKED_INPUT", "Caption cache schema/norms invalid")
    if not set(captions.caption_id) <= set(names):
        raise StopRun("BLOCKED_INPUT", "Caption embeddings incomplete")
    banks, visual = load_banks(root / "visual/features.npz")
    if not set(train.image) <= set(visual):
        raise StopRun("BLOCKED_INPUT", "Visual feature coverage incomplete")
    provenance = json.loads((root / "visual/provenance.json").read_text())
    for name in ("clip", "siglip2"):
        if provenance["encoders"][name]["revision"] != config["pretrained"][name]:
            raise StopRun("BLOCKED_INPUT", "Frozen visual checkpoint revision differs")
    if provenance["clip_large"]["revision"] != config["pretrained"]["clip_large"]:
        raise StopRun("BLOCKED_INPUT", "Frozen CLIP-L revision differs")
    hashes = json.loads((root / "visual/image_hashes.json").read_text())
    if not train.image.map(hashes).eq(train.image_sha256).all() or set(hashes) != set(visual):
        raise StopRun("BLOCKED_INPUT", "Image identity / cached-feature mapping differs")
    proof = reconstruct_history(root, train)
    return train, captions, dict(zip(names, values)), banks, visual, protocol, proof


def read_confirmation(root, protocol, *, cv_verified, authorized, final_models_verified, execution_scope=None):
    if execution_scope != "COMPETITION_READY" or not (cv_verified and authorized and final_models_verified):
        raise StopRun("BLOCKED_CONFIRMATION", "Confirmation remains sealed until verified CV_GO, advance authorization and final models")
    # Only this guarded function parses the opaque training-only historical label mapping.
    mapping = read_csv(Path(root) / "sealed/original_training_annotation_mapping.csv")
    locked = protocol[protocol.role.eq("holdout")]
    mapping = mapping[mapping.painting.isin(locked.painting)]
    mapping = mapping[mapping.emotion.isin(data.POSITIVE | data.NEGATIVE)]
    grouped = mapping.assign(positive=mapping.emotion.isin(data.POSITIVE)).groupby(data.KEYS).agg(n=("emotion", "size"), k=("positive", "sum")).reset_index()
    grouped = grouped[grouped.n.ge(5)]
    return grouped.merge(locked, on="painting", validate="many_to_one").assign(split="train")
