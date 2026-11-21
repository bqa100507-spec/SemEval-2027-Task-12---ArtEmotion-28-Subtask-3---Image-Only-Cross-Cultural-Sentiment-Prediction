"""Fit one culture-conditioned semantic head and a matched visual-semantic head."""

import re
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .baseline import file_sha256
from .cli import json_output
from .competition_challenger import check_hash, component_predictions, frame_sha
from .competition_ridge import fit_head, make_panel
from .competition_semantic_generation import CONFIG, cache_identity, read_record, training_inventory
from .competition_stack import ENCODERS, aligned_values, compare
from .competition_track1 import load_p2, peak_memory_bytes, save_csv
from .contrast_study import new_output
from .data import KEYS
from .e50_heads import read_json
from .metrics import painting_scores, score

ARMS = ("corrected_ridge", "semantic", "visual_semantic")


def field_terms(item):
    for field, value in item.items():
        tokens = re.findall(r"[a-z0-9]+", value.lower())
        for word in tokens:
            yield f"{field}:{word}"
        for left, right in zip(tokens, tokens[1:]):
            yield f"{field}:{left}_{right}"


def semantic_features(fit, query, config):
    if set(fit.index) & set(query.index):
        raise ValueError("Semantic vocabulary and projection require disjoint fit and query paintings")
    rule = config["predictor"]
    vocabulary = TfidfVectorizer(analyzer=field_terms, min_df=rule["tfidf_min_df"],
                                 max_features=rule["tfidf_max_features"], sublinear_tf=True)
    fitting_matrix = vocabulary.fit_transform(fit.tolist())
    # The guard mainly supports small offline fixtures. P2 fixes a 256-dimensional descriptor.
    size = min(rule["svd_components"], fitting_matrix.shape[0] - 1, fitting_matrix.shape[1] - 1)
    if size < 2:
        raise ValueError("Too few fitting features for a semantic projection")
    projection = TruncatedSVD(n_components=size, n_iter=7, random_state=rule["svd_seed"])
    train = pd.DataFrame(normalize(projection.fit_transform(fitting_matrix)), index=fit.index)
    held = pd.DataFrame(normalize(projection.transform(vocabulary.transform(query.tolist()))), index=query.index)
    return pd.concat([train, held]), {"vocabulary_size": fitting_matrix.shape[1], "dimensions": size,
                                    "zero_query_vectors": int(held.abs().sum(axis=1).eq(0).sum())}


def predict_head(train, keys, vectors, track1):
    if list(keys.columns) != KEYS or set(train.painting) & set(keys.painting):
        raise ValueError("Semantic prediction requires disjoint painting/language keys only")
    panel = make_panel(train)
    visual = np.ascontiguousarray(vectors.loc[list(panel.paintings)].to_numpy(dtype=np.float64))
    head, receipt = fit_head(panel, visual, track1["ridge"])
    frame = component_predictions({"p_positive": head}, keys, {"p_positive": vectors})
    return frame, receipt


def sealed_descriptions(args, config, track1):
    scope = training_inventory(args.inputs, track1)
    receipt = args.run_dir / "extraction.json"
    if not receipt.is_file():
        directory = args.run_dir / "descriptions"
        available = sum((directory / f"{digest}.json").is_file() for digest in scope.image_sha256)
        raise FileNotFoundError(f"Semantic CV needs a sealed cache for all {len(scope)} P2 paintings. {available} present.")
    manifest = read_json(receipt)
    if manifest["identity"] != cache_identity(config) or set(manifest["images"]) != set(scope.image_sha256):
        raise ValueError("Extraction seal does not match the frozen recipe or full P2 coverage")
    result = {}
    for row in scope.itertuples(index=False):
        path = args.run_dir / "descriptions" / f"{row.image_sha256}.json"
        check_hash(path, manifest["images"][row.image_sha256])
        result[row.painting] = read_record(path, row.image_sha256, config)["description"]
    return pd.Series(result), file_sha256(receipt)


def exact_reference(gold, path, config):
    check_hash(path / "oof.csv", config["incumbent_oof_sha256"])
    check_hash(path / "metrics.json", config["incumbent_metrics_sha256"])
    raw = pd.read_csv(path / "oof.csv")
    aligned_values(gold, raw, ["corrected"])
    joined = gold[KEYS + ["fold"]].merge(raw[KEYS + ["fold", "corrected"]], on=KEYS, validate="one_to_one",
                                         suffixes=("", "_reference"))
    if not joined.fold.eq(joined.fold_reference).all():
        raise ValueError("Corrected Ridge OOF data use different outer folds")
    return joined[KEYS].assign(corrected_ridge=joined.corrected)


def evaluate(frame, config, excluded):
    report = {}
    for scope, block in (("full", frame), ("filtered", frame.loc[~frame.language.isin(excluded)])):
        def score_part(part, arm):
            return score(part[KEYS + ["k", "n"]], part[KEYS].assign(p_positive=part[arm]))

        pooled = {arm: score_part(block, arm) for arm in ARMS}
        folds = {int(fold): {arm: score_part(part, arm) for arm in ARMS} for fold, part in block.groupby("fold")}
        table = block.assign(p=block.k / block.n)
        paintings = {arm: painting_scores(table, prediction=arm).set_index("painting").contrast for arm in ARMS}
        paired = {}
        for arm in ARMS[1:]:
            d = (paintings[arm] - paintings[ARMS[0]]).dropna()
            counts = table.groupby("painting").p.transform("size")
            include = table.painting.isin(d.index)
            centered = (table[["p", ARMS[0], arm]] - table.groupby("painting")[["p", ARMS[0], arm]].transform("mean"))
            e = (centered.p - centered[ARMS[0]])[include]
            f = (centered.p - centered[arm])[include]
            weight = 1 / counts[include]
            avg = lambda value: float(np.average(value, weights=weight))
            scale = (avg(e**2) * avg(f**2)) ** .5
            paired[arm] = compare(d, config["decision"]) | {
                "fold_wins": sum(x[arm]["cultural_contrast"] > x[ARMS[0]]["cultural_contrast"] for x in folds.values()),
                "log_loss_delta": pooled[arm]["log_loss"] - pooled[ARMS[0]]["log_loss"],
                "centered_residual_correlation": avg(e * f) / scale if scale else None}
        report[scope] = {"pooled": pooled, "folds": folds, "paired_vs_corrected_ridge": paired}
    rule = config["decision"]
    full, filtered = [report[s]["paired_vs_corrected_ridge"]["visual_semantic"] for s in ("full", "filtered")]
    passed = (full["contrast_delta"] >= rule["min_full_gain"] and filtered["contrast_delta"] >= rule["min_filtered_gain"]
              and filtered["bootstrap_95"][0] > 0 and filtered["fold_wins"] >= rule["min_fold_wins"]
              and filtered["trimmed_contrast_delta"] > 0
              and all(a["log_loss_delta"] <= rule["max_log_loss_delta"] for a in (full, filtered)))
    report["decision"] = "CV_GO_PENDING_ELIGIBILITY" if passed else "RETAIN CORRECTED RIDGE"
    return report


def run_cv(args, config, track1):
    start = time.perf_counter()
    destination = args.run_dir / "cv"
    if destination.exists():
        raise ValueError(f"Frozen CV output already exists: {destination}")
    descriptions, extraction_sha = sealed_descriptions(args, config, track1)
    gold, features, _ = load_p2(args.inputs, track1)
    reference = exact_reference(gold, args.reference_dir, config)
    visual = pd.concat([features[encoder] for encoder in ENCODERS], axis=1, ignore_index=True) / np.sqrt(3)
    decisions, fold_receipts = [], {}
    with threadpool_limits(limits=track1["threads"]):
        for fold, held in gold.groupby("fold"):
            tick = time.perf_counter()
            train = gold.loc[gold.fold.ne(fold)]
            fitting_ids, query_ids = sorted(train.painting.unique()), sorted(held.painting.unique())
            semantic, geometry = semantic_features(descriptions.loc[fitting_ids], descriptions.loc[query_ids], config)
            semantic_pred, b = predict_head(train, held[KEYS], semantic, track1)
            joint = pd.concat([visual, semantic], axis=1, ignore_index=True) / np.sqrt(2)
            visual_pred, c = predict_head(train, held[KEYS], joint, track1)
            baseline = aligned_values(held, reference.loc[held.index], [ARMS[0]]).ravel()
            blend_share = config["predictor"]["joint_share"]
            p_b = aligned_values(held, semantic_pred, ["p_positive"]).ravel()
            p_c = aligned_values(held, visual_pred, ["p_positive"]).ravel()
            decisions.append(held[KEYS].assign(corrected_ridge=baseline, semantic=p_b,
                                               visual_semantic=((1 - blend_share) * baseline + blend_share * p_c).round(6)))
            fold_receipts[int(fold)] = geometry | {"training_sha256": frame_sha(train),
                                                  "query_keys_sha256": frame_sha(held[KEYS]),
                                                  "semantic_fit": b, "joint_fit": c,
                                                  "seconds": time.perf_counter() - tick}
            print(f"VLM fold {fold}: B and C predicted {len(held)} held pairs", flush=True)
    outputs = gold.merge(pd.concat(decisions, ignore_index=True), on=KEYS, validate="one_to_one")
    metrics = evaluate(outputs, config, track1["test_like_excluded"])
    metrics |= {"fold_receipts": fold_receipts, "extraction_sha256": extraction_sha, "config_sha256": file_sha256(CONFIG),
                "cpu_wall_seconds": time.perf_counter() - start, "peak_rss_bytes": peak_memory_bytes(),
                "competition_eligibility": "UNRESOLVED"}
    with new_output(destination) as stage:
        save_csv(outputs[KEYS + ["fold", *ARMS]], stage / "oof.csv")
        json_output(metrics, stage / "metrics.json")
        json_output({file.name: file_sha256(file) for file in stage.iterdir()}, stage / "seal.json")
    print(metrics["decision"], flush=True)
