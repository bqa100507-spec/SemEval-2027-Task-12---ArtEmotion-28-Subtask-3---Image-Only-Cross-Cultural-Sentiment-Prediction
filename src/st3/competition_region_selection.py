"""Capacity-matched CPU readouts on frozen DINOv2-L spatial tokens."""

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F

from .baseline import file_sha256
from .cli import json_output
from .competition_challenger import frame_sha
from .competition_cultural_concepts import reference_rows
from .competition_ridge import make_panel
from .competition_semantic_generation import CONFIG as SEMANTIC
from .competition_stack import compare
from .competition_track1 import load_p2, peak_memory_bytes
from .contrast_study import new_output
from .data import KEYS, require_same_pairs
from .e50_heads import read_json
from .metrics import painting_scores, score

ARMS = ("global_pool", "shared_query", "language_query")
SYSTEMS = ("corrected_ridge", *ARMS, "blend")


class RegionHead(nn.Module):
    def __init__(self, dim, groups, query_width, arm):
        super().__init__()
        if arm not in ARMS or not 0 < query_width <= dim:
            raise ValueError("Invalid regional readout")
        self.arm, self.query_width = arm, query_width
        self.bias = nn.Parameter(torch.zeros(()))
        self.language = nn.Parameter(torch.zeros(groups))
        self.shared = nn.Parameter(torch.zeros(dim))
        self.deviation = nn.Parameter(torch.zeros(groups, dim))
        if arm != "global_pool":
            self.query = nn.Parameter(torch.zeros(query_width))

    def centered(self):
        return self.language - self.language.mean(), self.deviation - self.deviation.mean(0)

    def forward(self, x, *, force_shared=False):
        bias, dev = self.centered()
        slopes = self.shared + dev
        if self.arm == "global_pool":
            h = x.mean(1)
            return self.bias + bias + h @ slopes.T, None, self.bias + h @ self.shared
        keys = x[..., :self.query_width]
        common = self.query[None, :]
        query = common + dev[:, :self.query_width] if self.arm == "language_query" and not force_shared else common
        weights = (keys @ query.T * self.query_width ** .5).softmax(dim=1)
        pool = weights.transpose(1, 2) @ x
        logits = self.bias + bias + (pool * slopes).sum(-1)
        # Unknown target language uses the shared query and shared prediction weights.
        if query.shape[0] > 1:
            common_pool = ((keys @ common.T * self.query_width ** .5).softmax(dim=1).transpose(1, 2) @ x)[:, 0]
        else:
            common_pool = pool[:, 0]
        return logits, weights, self.bias + common_pool @ self.shared

    def regularizer(self, ridge, ratio, query_l2):
        bias, dev = self.centered()
        penalty = .5 * ridge * (bias.square().sum() + self.shared.square().sum() + ratio * dev.square().sum())
        if self.arm != "global_pool":
            penalty = penalty + .5 * query_l2 * self.query.square().sum()
        return penalty


def project(patches, fit_indices, config):
    rng = np.random.default_rng(config["seed"])
    chosen = np.sort(rng.choice(fit_indices, min(len(fit_indices), config["pca_paintings"]), replace=False))
    sample = np.asarray(patches[chosen, ::config["pca_patch_stride"]], dtype=np.float32)
    pca = PCA(n_components=config["pca_width"], svd_solver="randomized", iterated_power=3, random_state=config["seed"])
    pca.fit(sample.reshape(-1, sample.shape[-1]))
    result = np.empty((*patches.shape[:2], config["pca_width"]), np.float32)
    for i in range(0, len(patches), 24):
        cells = np.asarray(patches[i:i + 24], dtype=np.float32)
        shape = (*cells.shape[:2], config["pca_width"])
        transformed = pca.transform(cells.reshape(-1, cells.shape[-1])).reshape(shape)
        norm = np.linalg.norm(transformed, axis=-1, keepdims=True).clip(1e-8)
        result[i:i + 24] = transformed / norm
    return torch.from_numpy(result), pca


def train_arm(x, fit, config, rule, arm):
    model = RegionHead(x.shape[-1], len(fit.languages), config["query_width"], arm)
    k, n = (torch.as_tensor(v, dtype=torch.float32) for v in (fit.k, fit.n))
    average_votes = n.sum() / len(x)
    ridge = 1 / (rule["reference_c"] * n.sum())
    with torch.no_grad():
        model.bias.copy_(torch.logit((k.sum() + .5) / (n.sum() + 1)))
    opt = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    rng = np.random.default_rng(config["seed"] + 1)
    curve = []
    start = time.perf_counter()
    for _ in range(config["epochs"]):
        total = 0.0
        for rows in np.array_split(rng.permutation(len(x)), int(np.ceil(len(x) / config["paintings_per_batch"]))):
            logits, _, _ = model(x[rows])
            batch = (n[rows] * F.softplus(logits) - k[rows] * logits).sum()
            loss = batch / (len(rows) * average_votes) + model.regularizer(ridge, rule["deviation_ratio"], config["query_l2"])
            if not loss.isfinite():
                raise RuntimeError(f"Non-finite regularized binomial loss for {arm}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(batch.detach())
        curve.append(total / float(n.sum()))
    return model.eval(), {"train_bce_nats": curve, "seconds": time.perf_counter() - start,
                          "parameters": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def predict(model, x, names, groups, keys):
    idx, langs = pd.Index(names).get_indexer(keys.painting), pd.Index(groups).get_indexer(keys.language)
    if (idx < 0).any() or list(keys.columns) != KEYS:
        raise ValueError("Spatial prediction requires painting/language keys only")
    out, profiles = np.empty(len(keys)), []
    for start in range(0, len(names), 32):
        logits, weights, fallback = model(x[start:start + 32])
        hit = (idx >= start) & (idx < start + len(logits))
        i = idx[hit] - start
        out[hit] = torch.cat([logits, fallback[:, None]], 1)[i, langs[hit]].sigmoid().numpy()
        if weights is not None:
            profiles.extend(weights.numpy())
    return out.round(6), profiles


def attention_audit(profiles, names, groups, keys):
    records, samples = [], []
    mapped = keys.groupby("painting").language.apply(list)
    chosen = set(sorted(names, key=lambda p: hashlib.sha256(p.encode()).digest())[:2])
    for name, weights in zip(names, profiles, strict=True):
        labs = [l for l in mapped[name] if l in groups]
        if len(labs) < 2:
            continue
        w = weights[:, [groups.index(l) for l in labs]].T
        i, j = np.triu_indices(len(w), 1)
        diff = .5 * np.abs(w[i] - w[j]).sum(1)
        records.append({"painting": name, "mean_tv": float(diff.mean()),
                        "top_patch_change": float(np.mean(w[i].argmax(1) != w[j].argmax(1)))})
        if name in chosen:
            samples.append({"painting": name, "grid": [16, 16], "languages": labs, "weights": w.tolist()})
    return records, samples


def evaluate(gold, output, config, track):
    require_same_pairs(gold, output)
    frame = gold.merge(output, on=KEYS, validate="one_to_one").assign(p=lambda d: d.k / d.n)
    report = {}
    for scope, part in (("full", frame), ("test_like", frame[~frame.language.isin(track["test_like_excluded"])])):
        def scorer(d, model):
            values = score(d[KEYS + ["k", "n"]], d[KEYS].assign(p_positive=d[model]))
            return {k: values[k] for k in ("log_loss", "cultural_contrast", "contrast_paintings")}

        pooled = {a: scorer(part, a) for a in SYSTEMS}
        folds = {int(i): {a: scorer(g, a) for a in SYSTEMS} for i, g in part.groupby("fold")}
        paintings = {a: painting_scores(part, a).set_index("painting").contrast for a in SYSTEMS}
        pairs = {}
        for a, b in [("shared_query", "global_pool"), ("language_query", "shared_query"),
                     ("language_query", "corrected_ridge"), ("blend", "corrected_ridge")]:
            d = (paintings[a] - paintings[b]).dropna()
            wins = min(int(np.ceil(len(d) * .01)), int(d.gt(0).sum()))
            changes = {i: row[a]["cultural_contrast"] - row[b]["cultural_contrast"] for i, row in folds.items()}
            pairs[f"{a}_minus_{b}"] = compare(d, track["decision"]) | {
                "fold_deltas": changes, "fold_wins": sum(v > 1e-12 for v in changes.values()),
                "drop_top_wins": float(d.sort_values(ascending=False).iloc[wins:].mean()),
                "loss_delta": pooled[a]["log_loss"] - pooled[b]["log_loss"]}
        report[scope] = {"pooled": pooled, "folds": folds, "paired": pairs, "pairs": len(part)}
    c = report["test_like"]["paired"]["language_query_minus_shared_query"]
    selected = []
    for a in ("language_query", "blend"):
        f, t = (report[s]["paired"][f"{a}_minus_corrected_ridge"] for s in ("full", "test_like"))
        if (c["bootstrap_95"][0] > 0 and c["trimmed_contrast_delta"] > 0 and f["contrast_delta"] >= 0
                and t["contrast_delta"] >= config["min_gain"] and t["bootstrap_95"][0] > 0
                and t["fold_wins"] >= config["min_wins"] and t["trimmed_contrast_delta"] > 0 and t["drop_top_wins"] > 0
                and max(f["loss_delta"], t["loss_delta"]) <= config["max_log_loss_delta"]):
            selected.append(a)
    report["selected"] = max(selected, key=lambda a: report["test_like"]["pooled"][a]["cultural_contrast"],
                             default="corrected_ridge")
    return report


def run(args, names, patches, meta, config, track):
    start = time.perf_counter()
    torch.set_num_threads(config["cpu_threads"])
    gold, _, _ = load_p2(args.inputs, track)
    if len(names) != len(set(names)) or set(names) != set(gold.painting):
        raise ValueError("Expected the exact pinned P2 artwork population")
    lookup = pd.Index(names)
    reference = reference_rows(gold, args.reference, read_json(SEMANTIC))
    pieces, fits, audits, maps = [], {}, [], []
    with threadpool_limits(limits=config["cpu_threads"]):
        for fold, held in gold.groupby("fold"):
            start_fold = time.perf_counter()
            train = gold.loc[gold.fold.ne(fold)]
            panel = make_panel(train)
            held_names = sorted(held.painting.unique())
            if set(held_names) & set(panel.paintings):
                raise ValueError("Painting leakage across the original fold")
            x, pca = project(patches, lookup.get_indexer(panel.paintings), config)
            xt, xh = x[lookup.get_indexer(panel.paintings)], x[lookup.get_indexer(held_names)]
            row, keys = {}, held[KEYS]
            out = keys.copy()
            for arm in ARMS:
                head, row[arm] = train_arm(xt, panel, config, track["ridge"], arm)
                tick = time.perf_counter()
                out[arm], profiles = predict(head, xh, held_names, panel.languages, keys)
                row[arm]["inference_seconds"] = time.perf_counter() - tick
                if arm == "language_query":
                    records, examples = attention_audit(profiles, held_names, panel.languages, keys)
                    audits.extend(r | {"fold": int(fold)} for r in records)
                    maps.extend(r | {"fold": int(fold)} for r in examples)
            if row["shared_query"]["parameters"] != row["language_query"]["parameters"]:
                raise ValueError("Shared and target-language attention parameter counts differ")
            pieces.append(out)
            fits[int(fold)] = row | {"fit_sha256": frame_sha(train), "held_keys_sha256": frame_sha(keys),
                                     "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
                                     "seconds": time.perf_counter() - start_fold}
            print(f"Fold {fold} complete in {fits[int(fold)]['seconds']:.1f}s (CPU)", flush=True)
    out = reference.drop(columns="fixed_ridge").merge(pd.concat(pieces), on=KEYS, validate="one_to_one")
    share = config["blend_share"]
    out["blend"] = ((1 - share) * out.corrected_ridge + share * out.language_query).round(6)
    result = evaluate(gold, out, config, track)
    result |= {"fits": fits, "wall_seconds": time.perf_counter() - start, "peak_rss_bytes": peak_memory_bytes(),
               "patch_receipt": meta, "source_sha256": file_sha256(Path(__file__)), "config": config,
               "attention": pd.DataFrame(audits)[["mean_tv", "top_patch_change"]].mean().to_dict()}
    with new_output(args.run_dir / "results") as stage:
        gold[KEYS + ["fold"]].merge(out, on=KEYS, validate="one_to_one").to_csv(stage / "oof.csv", index=False, float_format="%.6f")
        pd.DataFrame(audits).to_csv(stage / "attention_audit.csv", index=False)
        json_output(maps, stage / "attention_maps.json")
        json_output(result, stage / "metrics.json")
        json_output({f.name: file_sha256(f) for f in stage.iterdir()}, stage / "seal.json")
    print(f"P2 choice: {result['selected']}", flush=True)
