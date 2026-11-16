"""Standalone vote-weighted visual neighbors; no E50 predictions or text inputs."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data


def unit_rows(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Features must be a matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or (norms <= 0).any():
        raise ValueError("Features must be finite nonzero image vectors")
    return values / norms


def visual_features(large, siglip):
    """Equal cosine contribution from the two existing, frozen image encoders."""
    if np.shape(large) != np.shape(siglip) or np.shape(large)[-1] != 768:
        raise ValueError("Expected aligned CLIP-L/SigLIP2 768-dimensional blocks")
    return np.concatenate([unit_rows(large), unit_rows(siglip)], axis=1) / np.sqrt(2.)


@dataclass(frozen=True)
class Parameters:
    neighbors: int
    temperature: float
    shrinkage: float

    def __post_init__(self):
        if (not isinstance(self.neighbors, int) or self.neighbors < 1
                or not np.isfinite([self.temperature, self.shrinkage]).all()
                or self.temperature <= 0 or self.shrinkage <= 0):
            raise ValueError("Neighbors, temperature and shrinkage must be positive")


class RetrievalMemory:
    def __init__(self, training, features, prior_votes=20.):
        required = [*data.KEYS, "image", "n", "k", "split"]
        if training.empty or not set(required) <= set(training):
            raise ValueError("Training memory needs eligible original training vote counts")
        if training[data.KEYS].duplicated().any() or not training.split.eq("train").all():
            raise ValueError("Memory requires unique training-only painting-language pairs")
        counts = training[["n", "k"]].to_numpy(dtype=float)
        if (not np.isfinite(counts).all() or not np.equal(counts, np.floor(counts)).all()
                or (training.n < 5).any() or (training.k < 0).any() or (training.k > training.n).any()):
            raise ValueError("Invalid eligible binary vote counts")
        if not np.isfinite(prior_votes) or prior_votes <= 0:
            raise ValueError("Prior strength must be positive and finite")
        images = training[["painting", "image"]].drop_duplicates().sort_values("painting")
        if images.painting.duplicated().any() or images.image.duplicated().any():
            raise ValueError("Painting/image mapping is not one-to-one")
        self.paintings = tuple(images.painting)
        self.images = tuple(images.image)
        self.languages = tuple(sorted(training.language.unique()))
        self.features = unit_rows(np.stack([features[name] for name in self.images]))
        index = pd.MultiIndex.from_product([self.paintings, self.languages], names=data.KEYS)
        votes = training.set_index(data.KEYS)[["k", "n"]].reindex(index, fill_value=0)
        self.k = votes.k.to_numpy(dtype=float).reshape(len(images), -1)
        self.n = votes.n.to_numpy(dtype=float).reshape(len(images), -1)
        self.total = self.n.sum(axis=1)
        self.mean = self.k.sum(axis=1) / self.total
        self.global_prior = float(self.k.sum() / self.n.sum())
        self.prior = (self.k.sum(axis=0) + prior_votes * self.global_prior) / (self.n.sum(axis=0) + prior_votes)
        self.residual_votes = self.k - self.n * self.mean[:, None]

    def neighbors(self, queries, count):
        query = unit_rows(queries)
        if query.shape[1] != self.features.shape[1]:
            raise ValueError("Query and memory feature dimensions differ")
        cosine = np.clip(query @ self.features.T, -1., 1.)
        # Stable ties in canonical memory order; identifiers are never features.
        indices = np.argsort(-cosine, axis=1, kind="stable")[:, :min(count, len(self.paintings))]
        return indices, np.take_along_axis(cosine, indices, axis=1)

    def predict(self, query_features, languages, variant, parameters, *, neighbors=None):
        """Only pixel-derived vectors and requested language names enter inference."""
        if variant not in ("prior", "uncentered", "centered"):
            raise ValueError("Unknown retrieval variant")
        indices, cosine = self.neighbors(query_features, parameters.neighbors) if neighbors is None else neighbors
        indices, cosine = indices[:, :parameters.neighbors], cosine[:, :parameters.neighbors]
        weights = np.exp((cosine - 1.) / parameters.temperature)
        denominator = np.einsum("qj,qjl->ql", weights, self.n[indices])
        if variant == "prior":
            prediction = np.broadcast_to(self.prior, (len(indices), len(self.languages)))
        elif variant == "uncentered":
            positive = np.einsum("qj,qjl->ql", weights, self.k[indices])
            prediction = (positive + parameters.shrinkage * self.prior) / (denominator + parameters.shrinkage)
        else:
            total = np.einsum("qj,qj->q", weights, self.total[indices])
            positive = np.einsum("qj,qj->q", weights, self.total[indices] * self.mean[indices])
            image_mean = (positive + parameters.shrinkage * self.global_prior) / (total + parameters.shrinkage)
            residual = np.einsum("qj,qjl->ql", weights, self.residual_votes[indices])
            deviation = (residual + parameters.shrinkage * (self.prior - self.global_prior)) / (denominator + parameters.shrinkage)
            prediction = image_mean[:, None] + deviation
        fallback = image_mean if variant == "centered" else np.full(len(indices), self.global_prior)
        columns = {name: i for i, name in enumerate(self.languages)}
        output = np.column_stack([prediction[:, columns[name]] if name in columns else fallback for name in languages])
        return np.clip(output, 0., 1.)


def predict_pairs(memory, inputs, features, variant, parameters, *, neighbors=None):
    """IDs locate vectors and check exclusion; labels in an input frame are ignored."""
    inputs = inputs[data.INPUT_COLUMNS].copy()
    if inputs[data.KEYS].duplicated().any():
        raise ValueError("Duplicate query pairs")
    unique = inputs[["painting", "image"]].drop_duplicates().sort_values("painting")
    if unique.painting.duplicated().any():
        raise ValueError("Ambiguous query artwork")
    if set(unique.painting) & set(memory.paintings) or set(unique.image) & set(memory.images):
        raise ValueError("Query paintings/images overlap retrieval memory")
    languages = sorted(inputs.language.unique())
    vectors = np.stack([features[name] for name in unique.image])
    values = memory.predict(vectors, languages, variant, parameters, neighbors=neighbors)
    paintings = {name: i for i, name in enumerate(unique.painting)}
    columns = {name: i for i, name in enumerate(languages)}
    q = [values[paintings[p], columns[l]] for p, l in inputs[data.KEYS].itertuples(index=False, name=None)]
    return inputs[data.KEYS].assign(p_positive=np.round(q, 6))
