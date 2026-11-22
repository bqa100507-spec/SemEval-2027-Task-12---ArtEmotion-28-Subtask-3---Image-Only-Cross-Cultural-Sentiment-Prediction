import numpy as np
import pandas as pd
import torch

from st3.competition_region_selection import ARMS, RegionHead, attention_audit, predict, project


def test_query_conditioning_has_equal_capacity_and_same_initial_model():
    x = torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(21))
    a, b, c = [RegionHead(8, 3, 3, arm) for arm in ARMS]
    for model in (a, b, c):
        with torch.no_grad():
            model.shared.fill_(0.2)
    assert sum(p.numel() for p in b.parameters()) == sum(p.numel() for p in c.parameters())
    torch.testing.assert_close(a(x)[0], b(x)[0])
    torch.testing.assert_close(b(x)[0], c(x)[0])
    with torch.no_grad():
        b.deviation[0, 0] = c.deviation[0, 0] = 1.0
    assert b(x)[1].shape[-1] == 1
    assert not torch.equal(c(x)[1][:, :, 0], c(x)[1][:, :, 1])
    torch.testing.assert_close(b(x)[0], c(x, force_shared=True)[0])
    keys = pd.DataFrame([("x", "Thai"), ("x", "Hindi"), ("y", "Unknown")], columns=["painting", "language"])
    probabilities, weights = predict(c, x, ["x", "y"], ["Thai", "Hindi", "Korean"], keys)
    assert np.isfinite(probabilities).all()
    audit, _ = attention_audit(weights, ["x", "y"], ["Thai", "Hindi", "Korean"], keys)
    assert len(audit) == 1 and audit[0]["mean_tv"] > 0


def test_patch_projection_is_fitted_only_on_training_rows():
    x = np.random.default_rng(15).normal(size=(7, 6, 8)).astype(np.float16)
    config = {"seed": 4, "pca_width": 3, "pca_paintings": 5, "pca_patch_stride": 2}
    fit = np.arange(5)
    first, pca1 = project(x, fit, config)
    x[5:] = 99
    second, pca2 = project(x, fit, config)
    np.testing.assert_array_equal(pca1.components_, pca2.components_)
    torch.testing.assert_close(first[:5], second[:5])
