"""Grouped training and state restoration for EXP-033."""
import copy
import hashlib
import math
import random
from time import monotonic

import numpy as np
import torch

from .data import KEYS
from .exp033 import Adapter, SpatialHead, drift_sum, suffix, vote_sum


def deterministic_order(names, seed):
    return sorted(names, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).digest())


def inner_split(train, fraction, seed):
    names = deterministic_order(train.painting.unique(), seed)
    held = set(names[:round(len(names) * fraction)])
    mask = train.painting.isin(held)
    return train.loc[~mask], train.loc[mask]


class Panel:
    def __init__(self, frame, indices):
        self.names = sorted(frame.painting.unique())
        self.rows = [indices[name] for name in self.names]
        self.languages = sorted(frame.language.unique()) if "k" in frame else []
        self.rows_by_name = {name: i for i, name in enumerate(self.names)}
        if self.languages:
            self.k = torch.zeros(len(self.names), len(self.languages))
            self.n = torch.zeros_like(self.k)
            row = frame.painting.map(self.rows_by_name).to_numpy()
            col = frame.language.map({name: i for i, name in enumerate(self.languages)}).to_numpy()
            self.k[row, col] = torch.as_tensor(frame.k.to_numpy(), dtype=torch.float32)
            self.n[row, col] = torch.as_tensor(frame.n.to_numpy(), dtype=torch.float32)


class Bank:
    def __init__(self, arrays, device):
        self.arrays = arrays
        self.device = torch.device(device)

    def take(self, rows, fields=("tokens", "global", "mask", "shapes")):
        return {k: torch.as_tensor(self.arrays[k][rows]).to(self.device) for k in fields}


class Trainer:
    def __init__(self, vision, bank, panel, config):
        self.vision, self.bank, self.panel, self.config = vision, bank, panel, config
        self.device = bank.device
        self.cuda = self.device.type == "cuda"
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        self.head = SpatialHead(config["width"], panel.languages, config).to(self.device)
        self.adapters = torch.nn.ModuleList([
            Adapter(config["width"], config["adapter_width"]) for _ in range(config["tail_blocks"])
        ]).to(self.device)
        with torch.no_grad():
            self.head.intercept.copy_(torch.logit((panel.k.sum() + .5) / (panel.n.sum() + 1)).to(self.device))
        self.optimizer = torch.optim.AdamW([
            {"params": self.head.parameters(), "lr": config["head_lr"], "weight_decay": 0},
            {"params": self.adapters.parameters(), "lr": config["adapter_lr"], "weight_decay": config["adapter_decay"]},
        ], foreach=False)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cuda)
        self.probe = False
        self.fixed = None

    def groups(self, arm, epoch):
        attention = arm in ("C", "D")
        adapting = arm == "D" and epoch > self.config["warmup_epochs"] and not self.probe
        groups = {"head": [], "pool": [], "adapter": []}
        for name, parameter in self.head.named_parameters():
            pool = name in self.head.pool_state_names()
            parameter.requires_grad_(attention or not pool)
            groups["pool" if pool else "head"].append(parameter)
        for parameter in self.adapters.parameters():
            parameter.requires_grad_(adapting)
            groups["adapter"].append(parameter)
        return groups

    def forward(self, rows, arm, epoch):
        if self.fixed is not None:
            h = torch.stack([self.fixed[row] for row in rows]).to(self.device)
            return h, h.new_zeros(())
        adapted = self.probe or (arm == "D" and epoch > self.config["warmup_epochs"])
        data = self.bank.take(rows)
        tokens = data["tokens"]
        if adapted:
            pre = self.bank.take(rows, ("prefix",))["prefix"]
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.cuda):
                tokens = suffix(self.vision, pre, data["mask"], self.config["tail_blocks"], self.adapters)
        h = self.head.representation(tokens, data["global"], data["mask"], data["shapes"], arm)
        drift = drift_sum(tokens, data["tokens"], data["mask"]) if adapted and not self.probe else h.new_zeros(())
        return h, drift

    def train_epoch(self, arm, epoch, microbatch, *, deadline=float("inf")):
        panel, rule = self.panel, self.config
        step = rule["step_images"]
        total_votes = float(panel.n.sum())
        denominator = total_votes / math.ceil(len(panel.names) / step)
        order = [panel.rows_by_name[name] for name in deterministic_order(panel.names, f"{rule['seed']}/{epoch}")]
        groups = self.groups(arm, epoch)
        active = [p for params in groups.values() for p in params if p.requires_grad]
        sums = dict.fromkeys(groups, 0.)
        clipped = skipped = count = 0
        nll = drift = 0.
        if self.cuda:
            torch.cuda.synchronize(self.device)
        start = monotonic()
        for offset in range(0, len(order), step):
            if monotonic() >= deadline:
                raise TimeoutError("Session budget reached, restore the last published epoch")
            chunk = order[offset:offset + step]
            self.optimizer.zero_grad(set_to_none=True)
            for start_row in range(0, len(chunk), microbatch):
                rows = chunk[start_row:start_row + microbatch]
                h, change = self.forward([panel.rows[i] for i in rows], arm, epoch)
                logits = self.head.logits(h)
                vote = vote_sum(logits, panel.k[rows].to(self.device), panel.n[rows].to(self.device))
                loss = vote / denominator + rule["anchor"] * change / len(chunk)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite batch loss")
                self.scaler.scale(loss).backward()
                nll += float(vote.detach())
                drift += float(change.detach())
            ridge = self.head.ridge(1 / (rule["reference_c"] * total_votes), rule["deviation_ratio"])
            self.scaler.scale(ridge).backward()
            self.scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(active, rule["clip_norm"])
            norm_value = float(norm)
            clipped += norm_value > rule["clip_norm"]
            if math.isfinite(norm_value):
                # Measure the unscaled module norms from clipped grads, undoing the common clipping factor.
                inverse = max(1., (norm_value + 1e-6) / rule["clip_norm"])
                for name, params in groups.items():
                    per_parameter = [p.grad.detach().norm() for p in params if p.grad is not None]
                    sums[name] += float(torch.stack(per_parameter).norm()) * inverse if per_parameter else 0.
            elif not self.cuda:
                raise FloatingPointError("Nonfinite CPU gradient")
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            skipped += self.scaler.get_scale() < old_scale
            count += 1
        if self.cuda:
            torch.cuda.synchronize(self.device)
        duration = monotonic() - start
        return {"train_vote_nll": nll / total_votes, "drift": drift / len(panel.names),
                "mean_module_gradient_norms": {k: v / max(count - skipped, 1) for k, v in sums.items()},
                "clips": clipped, "steps": count, "skips": skipped,
                "clip_fraction": clipped / count, "skip_fraction": skipped / count,
                "seconds": duration, "paintings_per_second": len(panel.names) / duration,
                "trainable_parameters": {k: sum(p.numel() for p in v if p.requires_grad) for k, v in groups.items()}}

    @torch.no_grad()
    def predict(self, frame, indices, arm, epoch, microbatch):
        panel = Panel(frame[KEYS], indices)
        sources = {key: j for j, key in enumerate(self.panel.languages)}
        for begin in range(0, len(panel.rows), microbatch):
            rows = panel.rows[begin:begin + microbatch]
            h, _ = self.forward(rows, arm, epoch)
            known = self.head.logits(h).sigmoid().cpu().numpy()
            unknown = (self.head.intercept + h @ self.head.shared).sigmoid().cpu().numpy()
            if begin == 0:
                logits = np.empty((len(panel.rows), len(sources)))
                fallback = np.empty(len(panel.rows))
            logits[begin:begin + len(rows)] = known
            fallback[begin:begin + len(rows)] = unknown
        predicted = []
        for name, language in frame[KEYS].itertuples(index=False, name=None):
            i = panel.rows_by_name[name]
            predicted.append(logits[i, sources[language]] if language in sources else fallback[i])
        return frame[KEYS].assign(p_positive=np.asarray(predicted, dtype=float).round(6))

    def state(self, epoch, receipt):
        return {"head": self.head.state_dict(), "adapters": self.adapters.state_dict(),
                "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                "epoch": epoch, "receipt": receipt, "languages": self.panel.languages,
                "rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state(self.device) if self.cuda else None}

    def restore(self, checkpoint):
        if self.panel.languages != checkpoint["languages"]:
            raise ValueError("Checkpoint language map changed")
        self.head.load_state_dict(checkpoint["head"], strict=True)
        self.adapters.load_state_dict(checkpoint["adapters"], strict=True)
        self.optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        self.scaler.load_state_dict(checkpoint["scaler"])
        torch.set_rng_state(checkpoint["rng"])
        random.setstate(checkpoint["python_rng"])
        if self.cuda:
            torch.cuda.set_rng_state(checkpoint["cuda_rng"], self.device)


def select_epoch(c, d, tolerance):
    epochs = sorted(c)
    if epochs != sorted(d):
        raise ValueError("Missing matched inner checkpoints")
    losses = {e: (c[e]["log_loss"], d[e]["log_loss"]) for e in epochs}
    minima = [min(pair[i] for pair in losses.values()) for i in (0, 1)]
    eligible = [e for e in epochs if all(v <= m + tolerance for v, m in zip(losses[e], minima))]
    mean_contrast = lambda e: (c[e]["cultural_contrast"] + d[e]["cultural_contrast"]) / 2
    if eligible:
        return min(eligible, key=lambda e: (-mean_contrast(e), sum(losses[e]), e))
    return min(epochs, key=lambda e: (sum(losses[e]), -mean_contrast(e), e))
