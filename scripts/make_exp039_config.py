#!/usr/bin/env python3
"""Generate configs/exp039.json by code: EXP-038 pins, fresh Hugging Face pins for the 8B checkpoint, and the
hashes of the EXP-038 run-4 artifacts the overlay carries. Nothing in the output is typed by hand.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pin_hf_snapshot import pins  # noqa: E402
from st3.exp038_io import check_config_pins  # noqa: E402

QWEN_METADATA = ["config.json", "preprocessor_config.json", "video_preprocessor_config.json", "tokenizer_config.json",
                 "chat_template.jinja", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt"]
EXTRA_INSTRUCTIONS = ["Represent the subject matter and scene depicted in this artwork.",
                      "Represent the artistic style, colors and composition of this artwork.",
                      "Represent the mood and atmosphere of this artwork."]
CARRIED = {"run": "run.json", "p2_oof": "p2/oof.csv", "development_predictions": "development/predictions.csv",
           "cscva3_zip": "development/submissions/exp038_CSCVA_3_submission.zip",
           "incumbent_zip": "development/submissions/exp038_B2_C2_cscva3_submission.zip",
           **{f"feature_{arm}_{kind}": f"features/{arm}/{kind}" for arm in ("E1", "E2", "E3")
              for kind in ("features.npz", "seal.json")}}


def build(exp038, results_zip, e5_pins, e3_pins):
    config = copy.deepcopy(exp038)
    e3 = config["arms"]["E3"]
    if e3_pins["revision"] != e3["revision"] or e3_pins["weights_sha256"] != e3["weights_sha256"] \
            or e3_pins["metadata_git_blobs"] != e3["metadata_git_blobs"]:
        raise ValueError("Fresh 2B pins differ from the EXP-038 E3 pins")
    with ZipFile(results_zip) as z:
        carried = {key: hashlib.sha256(z.read(name)).hexdigest() for key, name in CARRIED.items()}
        selection = json.loads(z.read("selection.json"))
    if carried["incumbent_zip"] != selection["best_sha256"] or carried["cscva3_zip"] != selection["components_sha256"]["CSCVA_3"]:
        raise ValueError("EXP-038 results archive does not match its own selection")
    e5 = {key: value for key, value in e3.items() if key not in {"model", "revision", "weights_sha256",
                                                                  "metadata_git_blobs", "license", "width", "batch"}}
    e5 |= {"model": e5_pins["model"], "revision": e5_pins["revision"], "license": e5_pins["license"],
           "weights_sha256": e5_pins["weights_sha256"], "metadata_git_blobs": e5_pins["metadata_git_blobs"],
           "width": 4096, "batch": 4, "placement": "two_gpu", "max_memory_gib": 14,
           "single_gpu_fallback": {"placement": "nf4", "label": "E5_nf4"},
           "change_vs_E3": "model size only (FP16 split across both T4s; NF4 on one GPU, labelled E5_nf4)"}
    e6 = {key: value for key, value in e3.items() if key != "instruction"}
    e6 |= {"kind": "qwen3_vl_embedding_pool", "instructions": [e3["instruction"], *EXTRA_INSTRUCTIONS],
           "reuse_index_0": "E3", "feature": "L2-normalized mean of the 4 L2-normalized instruction embeddings",
           "change_vs_E3": "instruction pooling only"}
    e7 = copy.deepcopy(e3) | {"max_pixels": 1048576, "batch": 2, "change_vs_E3": "max_pixels 1,048,576 only"}
    config |= {
        "experiment": "EXP-039", "title": "Scale the VLM-embedding axis (size, instructions, resolution)",
        "owner_approval": "2026-11-27",
        "arms": {"E1": config["arms"]["E1"], "E2": config["arms"]["E2"], "E3": e3, "E5": e5, "E6": e6, "E7": e7},
        "new_arms": ["E5", "E6", "E7"],
        "candidates": {"C2": ["E1", "E2", "E3"], "D1": ["E1", "E2", "E5"], "D2": ["E1", "E2", "E6"],
                       "D3": ["E1", "E2", "E7"],
                       "D4": {"rule": "vlm_third", "clip_arms": ["E1", "E2"], "vlm_arms": ["E3", "E5", "E6", "E7"]}},
        "gated": ["D1", "D2", "D3", "D4"],
        "selection": {"rank": {"D2": 0, "D3": 0, "D1": 1, "D4": 2}, "tie_tolerance": 0.001,
                      "same_rank_tie": "lower measured GPU extraction seconds"},
        "gate": config["gate"] | {"reference": "C2", "minimum_filtered_gain": 0.003, "bootstrap_seed": 203904},
        "diagnostic_seed": 203905,
        "eligibility": {"vlm_free": ["R0_fixed", "R0_corrected", "E1", "E2"],
                        "unconfirmed_label": "UNCONFIRMED_QWEN_ST3"},
        "budget": config["budget"] | {"skip_order": ["E5", "E7"], "mandatory": ["R0", "C2_replay"]},
        "references": config["references"] | {
            "exp038_results_zip_sha256": hashlib.sha256(Path(results_zip).read_bytes()).hexdigest(),
            "exp038_carried_sha256": carried},
    }
    for key in ("cscva", "complexity_order", "similarity"):
        config.pop(key, None)
    config["similarity"] = {"candidates": exp038["similarity"]["candidates"], "R0": exp038["similarity"]["R0"]}
    check_config_pins(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp038-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "configs/exp039.json")
    args = parser.parse_args()
    exp038 = json.loads((ROOT / "configs/exp038.json").read_text(encoding="utf-8"))
    e5 = pins("Qwen/Qwen3-VL-Embedding-8B", "main", ["model-*.safetensors", "tokenizer.json"],
              [*QWEN_METADATA, "model.safetensors.index.json"], verify_download=True)
    e3 = exp038["arms"]["E3"]
    e3_fresh = pins(e3["model"], e3["revision"], ["model.safetensors", "tokenizer.json"], QWEN_METADATA,
                    verify_download=True)
    config = build(exp038, args.exp038_results, e5, e3_fresh)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: E5 {e5['model']}@{e5['revision']} ({e5['license']})")


if __name__ == "__main__":
    main()
