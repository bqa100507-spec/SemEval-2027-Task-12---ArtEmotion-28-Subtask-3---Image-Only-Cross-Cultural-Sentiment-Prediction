"""Matched shared eight-query auxiliary branch; inference is image/language only."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import data
from .caption_student import CaptionStudent, make_rows
from .gradient_analysis import diagnostic
from .polarity_student import balanced_channels, cosine_loss


class EmotionStudent(CaptionStudent):
    def __init__(self, languages, config):
        super().__init__(languages, config)
        width = config["hidden_width"]
        self.emotion = nn.Embedding(8, width)
        nn.init.normal_(self.emotion.weight, std=config["polarity_embedding_std"])
        self.auxiliary = nn.Sequential(nn.Linear(3*width, width), nn.GELU(),
                                       nn.Linear(width, config["target_dimensions"]))

    def forward(self, images, language, *, auxiliary=False):
        hidden = self.visual(images)
        known, index = language.ge(0), language.clamp_min(0)
        embedding = self.language_visual(index)*known[:, None]
        offset = self.language_bias(index).squeeze(-1)
        logits = self.shared(hidden).squeeze(-1)+torch.where(
            known, offset+(embedding*hidden).sum(-1)/self.scale, 0.)
        predicted = None
        if auxiliary:
            inputs = torch.cat([hidden[:, None].expand(-1, 8, -1), embedding[:, None].expand(-1, 8, -1),
                                self.emotion.weight[None].expand(len(hidden), -1, -1)], dim=-1)
            predicted = F.normalize(self.auxiliary(inputs), dim=-1)
        return logits, predicted


def seed_model(languages, seed, config, device):
    torch.manual_seed(seed)
    model = EmotionStudent(languages, config).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    return model, generator, optimizer


def emotion_rows(frame, features, languages, device, targets=None, weights=None):
    rows = make_rows(frame, features, languages, device)
    if targets is None:
        return rows
    example = next(iter(targets.values()))[0]
    empty = (np.zeros_like(example), np.zeros(8, dtype=bool))
    keys = list(frame[data.KEYS].itertuples(index=False, name=None))
    items = [targets.get(key, empty) for key in keys]
    rows["text"] = torch.as_tensor(np.stack([x[0] for x in items]), device=device)
    rows["text_mask"] = torch.as_tensor(np.stack([x[1] for x in items]), device=device)
    rows["text_weights"] = torch.tensor([weights.get(key, 0.) for key in keys], dtype=torch.float32, device=device)
    return rows


def query_matrix(values, masks):
    result, counts = np.full((8, 8), np.nan), np.zeros((8, 8), dtype=int)
    for left in range(8):
        for right in range(8):
            keep = masks[:, left] & masks[:, right]
            counts[left, right] = int(keep.sum())
            if keep.any():
                result[left, right] = np.clip((values[keep, left]*values[keep, right]).sum(axis=-1), -1., 1.).mean()
    return [[float(x) if np.isfinite(x) else None for x in row] for row in result], counts.tolist()


def emotion_diagnostic(model, rows, batch, config, collapse_threshold, *, require_nonzero=True):
    result = diagnostic(model, rows, batch, config["text_weight"], config["gradient_clip"])
    mode = model.training
    model.eval()
    captured = []
    hook = model.visual.register_forward_hook(lambda m, i, o: captured.append(o))
    try:
        batch = batch.to(rows["features"].device)
        _, prediction = model(rows["features"][rows["image"][batch]], rows["language"][batch], auxiliary=True)
        targets = rows["text"][batch].detach().clone().requires_grad_(True)
        masks, weights = rows["text_mask"][batch], rows["text_weights"][batch]
        loss = cosine_loss(prediction, targets, masks, weights)
        target_gradient, query_gradient = torch.autograd.grad(loss, (targets, model.emotion.weight), retain_graph=True)
        hidden = captured[0]
        effective = weights*masks.any(dim=1)
        denominator = effective.sum().clamp_min(1e-12)
        channel_losses = (1.-(prediction*targets).sum(-1).clamp(-1., 1.))*masks/masks.sum(1).clamp_min(1)[:, None]
        channel_gradients = []
        for channel in range(8):
            component = (channel_losses[:, channel]*effective).sum()/denominator
            channel_gradients.append(torch.autograd.grad(component, hidden, retain_graph=True)[0].detach().flatten())
        gradients = torch.stack(channel_gradients)
        norms = gradients.norm(dim=1)
        gradient_cosine = F.normalize(gradients, dim=1) @ F.normalize(gradients, dim=1).T
        query_norms = query_gradient.norm(dim=1)
        coverage = masks.sum(dim=0)
        if not bool(torch.isfinite(query_gradient).all()) or not bool(torch.isfinite(gradients).all()):
            raise ValueError("Nonfinite emotion query gradients")
        if require_nonzero and bool((query_norms[coverage > 0] <= 0).any()):
            raise ValueError("An available emotion query received no gradient")
        if bool((target_gradient[~masks] != 0).any()) or bool((query_norms[coverage == 0] != 0).any()):
            raise ValueError("Missing emotion channels contribute target gradients")
        predicted = prediction.detach().cpu().numpy()
        expected = targets.detach().cpu().numpy()
        keep = masks.cpu().numpy()
        pmatrix, counts = query_matrix(predicted, keep)
        tmatrix, _ = query_matrix(expected, keep)
        offdiag = ~np.eye(8, dtype=bool)
        fullcos = np.einsum("nkd,njd->nkj", predicted, predicted)
        collapsed_all = fullcos[:, offdiag].mean(axis=1) >= collapse_threshold
        collapsed_valid = []
        for cosine, valid in zip(fullcos, keep):
            pairs = valid[:, None] & valid[None, :] & offdiag
            if pairs.any():
                collapsed_valid.append(bool(cosine[pairs].mean() >= collapse_threshold))
        result.update({"coverage":coverage.tolist(), "query_embedding_gradient_norms":query_norms.tolist(),
            "available_zero_gradient_channels":torch.nonzero((coverage>0)&(query_norms==0)).flatten().tolist(),
            "per_emotion_shared_gradient_norms":norms.tolist(),
            "per_emotion_scaled_shared_gradient_norms":(config["text_weight"]*norms).tolist(),
            "emotion_gradient_cosine":[[float(value) if norms[i] > 0 and norms[j] > 0 else None
                                        for j, value in enumerate(row)] for i, row in enumerate(gradient_cosine)], "predicted_query_cosine":pmatrix,
            "target_query_cosine":tmatrix, "coobserved_counts":counts,
            "missing_target_gradient_max":float(target_gradient[~masks].abs().max()) if bool((~masks).any()) else 0.,
            "collapsed_all_query_fraction":float(collapsed_all.mean()),
            "collapsed_available_query_fraction":float(np.mean(collapsed_valid)) if collapsed_valid else None,
            "collapse_eligible_pairs":len(collapsed_valid)})
        return result
    finally:
        hook.remove()
        model.train(mode)
