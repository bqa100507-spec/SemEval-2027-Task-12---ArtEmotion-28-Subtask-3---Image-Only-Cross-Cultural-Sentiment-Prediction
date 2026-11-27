"""EXP-039: VLM-embedding scaling arms on top of the EXP-038 C2 recipe (pure CPU logic).

Everything statistical reuses ``st3.exp038`` unchanged: the Track 1 heads, the 0.5/0.5 mix, the
neighborhood bank and correction, the P2 report, gate arithmetic and the B1/B2 compositions.
"""

import numpy as np

from . import exp038 as science
from .data import KEYS

SEALED = ("E1", "E2", "E3")
NEW = ("E5", "E6", "E7")


def label(config_arm, e5_label):
    return e5_label if config_arm == "E5" else config_arm


def config_arm(arm):
    return "E5" if arm.startswith("E5") else arm


def arm_spec(config, arm):
    spec = dict(config["arms"][config_arm(arm)])
    if arm == "E5_nf4":
        spec |= {"placement": spec["single_gpu_fallback"]["placement"]}
    if config_arm(arm) in {"E5", "E6", "E7"}:
        # E5-E7 change exactly one property of E3; everything else (formatting, pixels, dtype) is E3's.
        spec = config["arms"]["E3"] | spec
    return spec


def candidate_set(config, arms, e5_label="E5"):
    """C2 and every D candidate whose arms completed; D4 needs at least one of E5-E7."""
    out = {}
    for name, spec in config["candidates"].items():
        if isinstance(spec, dict):
            vlm = tuple(label(a, e5_label) for a in spec["vlm_arms"] if label(a, e5_label) in arms)
            if not [a for a in vlm if a != "E3"] or not set(spec["clip_arms"]) <= set(arms):
                continue
            out[name] = science.VlmThird(tuple(spec["clip_arms"]), vlm)
        else:
            parts = tuple(label(a, e5_label) for a in spec)
            if set(parts) <= set(arms):
                out[name] = parts
    return out


def unavailable(config, candidates, pieces, e5_label="E5"):
    out = {}
    for name, spec in config["candidates"].items():
        if name in candidates:
            continue
        needed = spec["vlm_arms"][1:] if isinstance(spec, dict) else [a for a in spec if a in NEW]
        states = {label(a, e5_label): (pieces.get(label(a, e5_label)) or {}).get("state", "NOT_RUN") for a in needed}
        joined = "/".join(states) + " " + "/".join(states.values())
        out[name] = {"status": "NOT_AVAILABLE", "passed": False, "reason": f"NOT_AVAILABLE ({joined})",
                     "arm_states": states}
    return out


def pooled_instructions(e3, raw, instructions):
    """E6 = L2-normalize(mean of E3 and the extra-instruction embeddings), per image."""
    e3 = np.asarray(e3, dtype=np.float64)
    raw = np.asarray(raw, dtype=np.float64).reshape(len(e3), instructions - 1, e3.shape[1])
    mean = np.concatenate([e3[:, None, :], raw], axis=1).mean(axis=1)
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    if not np.isfinite(mean).all() or (norms <= 0).any():
        raise FloatingPointError("Invalid pooled E6 embedding")
    return (mean / norms).astype(np.float32)


def pool_items(images, instructions):
    """Extra E6 items: (image|k, path, k) for k = 1..instructions-1; index 0 is the sealed E3 feature."""
    return [(f"{digest}|{k}", path, k) for digest, path in images for k in range(1, instructions)]


def select(gates, config, gpu_seconds):
    """Highest filtered Contrast; within the tolerance the simpler rank; same rank by lower GPU cost."""
    rule = config["selection"]
    passing = [name for name, gate in gates.items() if gate.get("passed")]
    if not passing:
        return None
    best = max(gates[name]["numbers"]["filtered_contrast"] for name in passing)
    close = [name for name in passing if best - gates[name]["numbers"]["filtered_contrast"] <= rule["tie_tolerance"]]

    def cost(name):
        spec = config["candidates"][name]
        arms = spec["vlm_arms"] if isinstance(spec, dict) else spec
        return sum(gpu_seconds.get(a, 0.) for a in arms if config_arm(a) in NEW)

    return min(close, key=lambda name: (rule["rank"][name], cost(name), name))


def replay_columns(gold, mine, archived, columns, threshold, prefix):
    return [science.measure(f"{prefix} {column}", science.aligned_values(gold, mine, [column]).ravel(),
                            science.aligned_values(gold, archived, [column]).ravel(), threshold)
            for column in columns]


def keys_frame(frame):
    return frame[KEYS]
