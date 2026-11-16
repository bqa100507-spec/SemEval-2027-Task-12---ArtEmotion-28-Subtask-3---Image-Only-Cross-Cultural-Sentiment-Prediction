"""One visual review of matched NF4/FP16 smoke; warnings are not vetoes."""

import base64
from html import escape
import io
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image

from .vlm_semantics import (SemanticCache, atomic_json, file_hash, fingerprint,
                            semantic_warnings, smoke_sample, training_images)


class SmokeReviewRequired(RuntimeError):
    pass


class SystematicHallucination(RuntimeError):
    pass


def smoke_signature(cache, smoke):
    records = []
    for row in smoke.itertuples():
        record = cache.read(row.image_sha256)
        if not record or "embedding" not in record:
            raise ValueError("Incomplete or technically invalid smoke embeddings")
        records.append({key: record[key] for key in ("image_sha256", "description_sha256", "embedding_sha256")})
    return {"semantic_identity": fingerprint(cache.identity), "records": records}


def timing_summary(cache, records):
    """Never substitute cache-read wall time for actual generation measurements."""
    profiles, missing_profiles = [], []
    run_ids = {r.get("generation_run_id") for r in records}
    for run_id in run_ids - {None}:
        path = cache.directory / f"telemetry_{run_id}.json"
        if path.is_file():
            profiles.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing_profiles.append(run_id)  # Abrupt VM loss must not invalidate completed image features.
    if None in run_ids:
        # Historical NF4 records predate explicit run IDs. Keep the attribution
        # limitation visible instead of inventing a matched benchmark.
        for path in cache.directory.glob("telemetry_*.json"):
            profile = json.loads(path.read_text(encoding="utf-8"))
            if profile.get("generated") == len(records):
                profiles.append(profile)
    return {"GPU_names": sorted({r["GPU"] for r in records}), "images": len(records),
            "mean_generation_seconds": float(np.mean([r["seconds"] for r in records])),
            "median_generation_seconds": float(np.median([r["seconds"] for r in records])),
            "peak_allocated_bytes": max((r["peak_allocated"] for r in profiles), default=None),
            "peak_reserved_bytes": max((r["peak_reserved"] for r in profiles), default=None),
            "missing_generation_profiles": missing_profiles,
            "peak_scope": "available persisted generation profiles; null if unavailable",
            "timing_source": "Persisted actual generation records, not this cache-read attempt",
            "profile_attribution": "exact generation run IDs" if None not in run_ids else "historical30-image telemetry; no run IDs"}


def build_comparison(inputs, nf4_root, fp16_root, nf4_config, fp16_config, output):
    for key in ("prompt", "decoding", "text_encoder", "runtime", "smoke_seed", "smoke_images"):
        if nf4_config[key] != fp16_config[key]:
            raise ValueError("NF4/FP16 comparison must use the same images, prompt, decoding and text encoder")
    left_vlm, right_vlm = dict(nf4_config["vlm"]), dict(fp16_config["vlm"])
    if left_vlm.pop("quantization") != "nf4" or right_vlm.pop("quantization") != "fp16" or left_vlm != right_vlm:
        raise ValueError("Only VLM precision may differ in the smoke comparison")
    images = training_images(inputs, "869b2fe227fb614d83827d8124a80bd02c5f7677482bf64ae7098856b9c70eed")
    smoke = smoke_sample(images, fp16_config["smoke_images"], fp16_config["smoke_seed"])
    caches = [SemanticCache(nf4_root, nf4_config), SemanticCache(fp16_root, fp16_config)]
    signatures = [smoke_signature(cache, smoke) for cache in caches]
    records = [[cache.read(row.image_sha256) for row in smoke.itertuples()] for cache in caches]
    timings = [timing_summary(cache, rows) for cache, rows in zip(caches, records)]
    warnings = [semantic_warnings(smoke, rows) for rows in records]
    rows, html = [], ["<h3>EXP-025: same30 artworks, same fixed prompt</h3>",
        "<p>Review broadly useful visual semantics, not sentence perfection. Isolated hallucinations and nationality/artist wording are warnings. Reject only serious systematic scene hallucination.</p>",
        "<div style='max-height:700px;overflow:auto'><table border='1' cellpadding='6'><tr><th>Artwork</th><th>NF4</th><th>FP16</th></tr>"]
    warning_maps = [{w["image_sha256"]: w["reasons"] for w in values} for values in warnings]
    for index, image in enumerate(smoke.itertuples()):
        left, right = records[0][index], records[1][index]
        with Image.open(Path(inputs) / "artwork/images" / image.image) as artwork:
            thumbnail = artwork.convert("RGB");thumbnail.thumbnail((220, 180))
            buffer = io.BytesIO();thumbnail.save(buffer, format="JPEG")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        flags = [warning_maps[i].get(image.image_sha256, []) for i in range(2)]
        rows.append({"painting": image.painting, "image": image.image, "image_sha256": image.image_sha256,
                     "nf4_description": left["description"], "fp16_description": right["description"],
                     "nf4_warning_reasons": ";".join(flags[0]), "fp16_warning_reasons": ";".join(flags[1]),
                     "nf4_seconds": left["seconds"], "fp16_seconds": right["seconds"]})
        html.append(f"<tr><td><img src='data:image/jpeg;base64,{data}'/><br>{escape(image.painting)}</td>"
                    f"<td>{escape(left['description'])}<p>{escape('; '.join(flags[0]))}</p></td>"
                    f"<td>{escape(right['description'])}<p>{escape('; '.join(flags[1]))}</p></td></tr>")
    html.append("</table></div>")
    same_gpu = len(timings[0]["GPU_names"]) == 1 and timings[0]["GPU_names"] == timings[1]["GPU_names"]
    result = {"technical_status": "PASS", "image_count": len(smoke), "same_images_and_prompt": True,
              "NF4": timings[0], "FP16": timings[1], "semantic_quality_warnings": {"NF4": warnings[0], "FP16": warnings[1]},
              "fp16_smoke": signatures[1], "nf4_smoke": signatures[0],
              "NF4_over_FP16_latency_ratio": timings[0]["mean_generation_seconds"] / timings[1]["mean_generation_seconds"] if same_gpu else None,
              "performance_comparison": "same GPU model" if same_gpu else "different hardware; no precision speedup claim",
              "qualitative_assessment": "Awaiting one owner visual review; automated word flags are not a factual-quality verdict"}
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "smoke_comparison.csv", index=False, lineterminator="\n")
    (output / "smoke_comparison.html").write_text("\n".join(html), encoding="utf-8")
    atomic_json(output / "smoke_comparison.json", result)
    return result


def record_review(comparison, decision, path, note=""):
    if decision not in ("BROADLY_USABLE", "SYSTEMATIC_SCENE_HALLUCINATION"):
        raise ValueError("Choose broadly usable semantics or serious systematic scene hallucination")
    comparison = Path(comparison)
    report = json.loads(comparison.read_text(encoding="utf-8"))
    value = {"decision": decision, "note": note, "timestamp": time.time(), "fp16_smoke": report["fp16_smoke"],
             "comparison_sha256": file_hash(comparison), "criterion": "Broad visual semantics; isolated errors/wording are warnings, not rejection criteria"}
    path = Path(path)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError("Preserve existing review; do not overwrite its decision")
    atomic_json(path, value)
    return value


def require_approval(cache, smoke, review_file):
    if review_file is None or not Path(review_file).is_file():
        raise SmokeReviewRequired("One visual smoke review is required before full extraction")
    review = json.loads(Path(review_file).read_text(encoding="utf-8"))
    if review["fp16_smoke"] != smoke_signature(cache, smoke):
        raise SmokeReviewRequired("Review describes different smoke records; preserve it and review the current comparison")
    if review["decision"] == "SYSTEMATIC_SCENE_HALLUCINATION":
        raise SystematicHallucination("Owner rejected systematic scene hallucination; preserve smoke and consider direct VLM hidden states or a stronger model, without launching either")
    if review["decision"] != "BROADLY_USABLE":
        raise SmokeReviewRequired("No valid visual smoke assessment")
    return review
