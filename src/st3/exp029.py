"""Soft visual prefixes for the exact Qwen2.5-VL decoder."""
import numpy as np
import torch
from torch import nn

from .caption_data import inner_partition, require_isolation
from .exp028_cache import KEYS
from .vlm_semantics import fingerprint


def pilot_partitions(rows, parent, config):
    fold = parent["fold"]
    pool, outer = rows[rows.fold.ne(fold)], rows[rows.fold.eq(fold)]
    train, val = inner_partition(pool, parent, fold)
    selections = []
    for frame, count, offset, pin in (
        (train, 192, 0, "pilot_train_sha256"), (val, 48, 1, "pilot_validation_sha256"),
    ):
        names = sorted(frame.painting.unique(), key=lambda p: fingerprint([config["seed"] + offset, p]))[:count]
        if len(names) != count or fingerprint(sorted(names)) != config[pin]:
            raise ValueError("EXP-026 pilot population identity differs")
        selections.append(frame[frame.painting.isin(names)].copy())
    require_isolation(*selections, outer)
    return selections


def crossfit_partitions(train, val, folds, seed):
    require_isolation(train, val)
    names = sorted(train.painting.unique(), key=lambda p: fingerprint([seed, p]))
    if len(names) < folds or folds < 2:
        raise ValueError("Insufficient painting groups for cross-fitting")
    for i in range(folds):
        query = train.painting.isin(names[i::folds])
        fit, held = train[~query], train[query]
        require_isolation(fit, held, val)
        yield fit, held


def visual_vectors(blocks, rows):
    return {name: np.concatenate([blocks["CLIP-L"][name], blocks["SigLIP2-Base"][name]]) / np.float32(np.sqrt(2))
            for name in rows.image.unique()}


def shuffled_images(rows, seed):
    names = rows[["painting", "image"]].drop_duplicates().sort_values("painting").image.to_numpy()
    ordered = np.random.default_rng(seed).permutation(names)
    if len(ordered) < 2 or len(ordered) != len(set(ordered)):
        raise ValueError("Shuffle requires distinct artworks")
    return rows.image.map(dict(zip(ordered, np.roll(ordered, 1)))).tolist()


def prefix_mask_positions(text_mask, tokens):
    # Soft tokens have no pixel grid. Place them on the same 1D diagonal
    # of Qwen's T/H/W rotary coordinates used by ordinary text.
    prefix = text_mask.new_ones((len(text_mask), tokens))
    mask = torch.cat([prefix, text_mask], dim=1)
    positions = (mask.long().cumsum(dim=1) - 1).masked_fill(mask == 0, 0)
    return mask, positions.unsqueeze(0).expand(3, -1, -1), mask.long().sum(dim=1) - 1


class VisualCorrection(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.tokens = config["visual_tokens"]
        self.projector = nn.Sequential(nn.Linear(1536, config["projector_width"]), nn.GELU(),
                                       nn.Linear(config["projector_width"], self.tokens * hidden_size))
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.scale = config["residual_scale"]

    def states(self, decoder, pixels, ids, text_mask):
        text = decoder.get_input_embeddings()(ids)
        visual = self.projector(pixels).reshape(len(ids), self.tokens, text.shape[-1]).to(text.dtype)
        embeds = torch.cat([visual, text], dim=1)
        mask, positions, final = prefix_mask_positions(text_mask, self.tokens)
        hidden = decoder(inputs_embeds=embeds, attention_mask=mask, position_ids=positions,
                         use_cache=False, return_dict=True).last_hidden_state
        return hidden[torch.arange(len(ids), device=ids.device), final]

    def forward(self, decoder, pixels, ids, text_mask, baseline):
        # Promote baseline logits and learned correction before adding them.
        return torch.logit(baseline.float()) + self.scale * self.head(
            self.states(decoder, pixels, ids, text_mask).float()).squeeze(-1)


def baseline_frame(frame, predictions):
    result = frame.merge(predictions, on=KEYS, validate="one_to_one")
    if len(result) != len(frame) or len(predictions) != len(frame):
        raise ValueError("Baseline predictions have incomplete coverage")
    if not result.p_positive.between(0, 1, inclusive="neither").all():
        raise ValueError("Rounded baseline contains nonfinite or boundary probabilities; logit is undefined")
    return result
