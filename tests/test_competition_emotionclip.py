import numpy as np
import pandas as pd
from PIL import Image
import pytest
import torch

from st3.competition_emotionclip import BRANCHES, evaluate
from st3.data import KEYS
from st3.emotionclip import EmotionCLIP, preprocess


def test_official_crop_uses_rounded_offset_and_rgb_normalization():
    image = Image.fromarray(np.broadcast_to(np.arange(227, dtype=np.uint8), (224, 227)).copy())
    output = preprocess(image)
    # 227 - 224 = 3, so torchvision places the left edge at round(1.5).
    raw = torch.tensor(2 / 255)
    torch.testing.assert_close(output[:, 0, 0], (raw - torch.tensor([.48145466, .4578275, .40821073]))
                               / torch.tensor([.26862954, .26130258, .27577711]))
    assert output.shape == (3, 224, 224)


def test_features_use_documented_clip_logits_and_unit_embeddings():
    model = EmotionCLIP.__new__(EmotionCLIP)
    model.used = set()
    model.state = {"logit_scale": torch.tensor(2.0).log()}
    model.encode_image = lambda _: torch.tensor([[6., 8.]])
    image, scores = model.features(None, torch.tensor([[2., 0.], [0., 8.]]))
    torch.testing.assert_close(image, torch.tensor([[.6, .8]]))
    torch.testing.assert_close(scores, torch.tensor([[1.2, 1.6]]).softmax(dim=-1))


def test_evaluation_recomputes_eligibility_and_enforces_coverage():
    gold = pd.DataFrame(
        [(f"p{i}", language, 1, k, 5) for i in range(20)
         for language, k in (("English", 4), ("Hindi", 1), ("Igbo", 2), ("Thai", 3))
         if i != 0 or language != "Thai"],
        columns=[*KEYS, "fold", "k", "n"])
    predictions = gold[KEYS].assign(**{name: gold.k / gold.n for name in
                                        ("fixed_ridge", "corrected_ridge", *BRANCHES, "blend")})
    track = {"test_like_excluded": ["English"], "decision": {"bootstrap_seed": 1, "bootstrap_draws": 10,
                                                            "trim_fraction": .05}}
    config = {"min_test_like_gain": .002, "min_fold_wins": 4, "max_loss_regression": .005}
    report, paintings = evaluate(gold, predictions, track, config)
    assert report["full"]["pooled"]["blend"]["contrast_paintings"] == 20
    assert report["test_like"]["pooled"]["blend"]["contrast_paintings"] == 19
    assert report["test_like"]["pooled"]["blend"]["log_loss_languages"] == 2
    assert report["selected"] == "corrected_ridge"
    assert len(paintings) == 39
    with pytest.raises(ValueError):
        evaluate(gold, predictions.iloc[:-1], track, config)
