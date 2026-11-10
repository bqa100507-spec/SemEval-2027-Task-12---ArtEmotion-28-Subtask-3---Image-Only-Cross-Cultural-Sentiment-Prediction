import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from st3.baseline import file_sha256
from st3.coverage import center_retained, dual_view, image_geometry, letterbox
from st3.coverage_study import DUAL, E50, FULL, choose_coverage, fold_reference, run_coverage_dev
from st3.visual_study import REFERENCE, SIGLIP


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "exp008.json").read_text())


def test_letterbox_keeps_all_four_edges_and_centers_padding():
    pad = CONFIG["padding_rgb"]
    image = Image.new("RGB", (8, 4), (30, 70, 90))
    for xy, color in (((0, 0), (255, 0, 0)), ((7, 0), (0, 255, 0)),
                      ((0, 3), (0, 0, 255)), ((7, 3), (255, 255, 0))):
        image.putpixel(xy, color)
    result = letterbox(image, 8, pad)
    assert result.size == (8, 8)
    np.testing.assert_array_equal(np.asarray(result)[2:6], np.asarray(image))
    assert result.getpixel((0, 1)) == result.getpixel((7, 6)) == tuple(pad)
    np.testing.assert_array_equal(np.asarray(letterbox(image.transpose(Image.Transpose.TRANSPOSE), 8, pad)),
                                  np.asarray(result.transpose(Image.Transpose.TRANSPOSE)))


def test_letterbox_uses_one_aspect_preserving_resize():
    result = np.asarray(letterbox(Image.new("L", (10, 30), 10), 18, CONFIG["padding_rgb"]))
    assert result.shape == (18, 18, 3)
    assert (result[:, 6:12, :] == 10).all()
    assert (result[:, :6, :] == CONFIG["padding_rgb"]).all()
    assert (result[:, 12:, :] == CONFIG["padding_rgb"]).all()


def test_dual_view_is_a_deterministic_unit_average():
    center = np.eye(2, dtype=np.float32)
    full = center[::-1]
    result = dual_view(center, full)
    np.testing.assert_allclose(result, np.full((2, 2), 1 / np.sqrt(2), dtype=np.float32))
    np.testing.assert_array_equal(dual_view(center, center), center)
    np.testing.assert_array_equal(dual_view(full, center), result)
    assert result.dtype == np.float32
    with pytest.raises(ValueError, match="nonzero"):
        dual_view(center, -center)


def test_geometry_uses_processor_rounding_and_needs_no_labels(tmp_path):
    assert center_retained(8, 3, 224) == pytest.approx(224 / 597)
    assert center_retained(3, 8, 224) == pytest.approx(224 / 597)
    assert center_retained(10, 10, 224) == 1
    folder = tmp_path / "images" / "style"
    folder.mkdir(parents=True)
    Image.new("RGB", (3, 8)).save(folder / "a.png")
    Image.new("RGB", (8, 8)).save(folder / "b.png")
    paintings = pd.DataFrame({"painting": ["a", "b"], "image_name": ["style/a.png", "style/b.png"],
                              "split": ["train", "val"]})
    geometry = image_geometry(tmp_path, paintings, CONFIG)
    assert geometry["orientation"].tolist() == ["portrait", "near-square"]
    assert geometry["substantially_cropped"].tolist() == [True, False]
    assert geometry["coverage_band"].tolist() == ["<65%", ">=95%"]
    assert geometry["style_folder"].tolist() == ["style", "style"]


def test_reference_reuses_identical_siglip_and_e50_predictions():
    keys = pd.DataFrame({"painting": ["a", "b"], "language": ["X", "Y"], "fold": 1})
    saved = keys.assign(p_C=[.2, .8])
    clip = keys.assign(model=REFERENCE, p_positive=[.2, .8])
    siglip = keys.assign(model=SIGLIP, p_positive=[.6, .4]).iloc[::-1]
    e50 = keys.assign(model=E50, p_positive=[.4, .6])
    baseline, reused = fold_reference(saved, siglip, pd.concat([clip, siglip, e50]), keys, 1, .5)
    assert baseline["p_positive"].tolist() == [.4, .6]
    assert reused["p_positive"].tolist() == [.4, .6]
    with pytest.raises(ValueError, match="Reference probabilities differ"):
        fold_reference(saved, siglip, pd.concat([clip, siglip, e50.assign(p_positive=.5)]), keys, 1, .5)
    with pytest.raises(ValueError, match="Pair mismatch"):
        fold_reference(saved, siglip, pd.concat([clip, siglip, e50]), keys.iloc[:1], 1, .5)


def selection_fixture():
    summary = pd.DataFrame({"pooled_contrast": [.44, .442, .443], "pooled_log_loss": [.72, .72, .72],
                            "log_loss_mean": [.73, .73, .73], "fold_wins": [0, 4, 5],
                            "converged": 5, "folds": 5, "scored": 5, "loss_scored": 5},
                           index=pd.Index([E50, FULL, DUAL], name="model"))
    return summary, {FULL: {"ci95": [.0001, .004]}, DUAL: {"ci95": [.0001, .005]}}


@pytest.mark.parametrize(("column", "value"), [("pooled_contrast", .4409), ("fold_wins", 3),
                                               ("pooled_log_loss", .731), ("log_loss_mean", .741),
                                               ("converged", 4), ("loss_scored", 4)])
def test_selection_rejects_a_candidate_if_any_gate_fails(column, value):
    summary, intervals = selection_fixture()
    assert choose_coverage(summary, intervals, CONFIG, 5)["model"] == DUAL
    summary.loc[DUAL, column] = value
    assert choose_coverage(summary, intervals, CONFIG, 5)["model"] == FULL


def test_no_go_preserves_development_outputs_and_rejects_tampered_cv(tmp_path):
    summary, intervals = selection_fixture()
    intervals[FULL]["ci95"][0] = 0.
    intervals[DUAL]["ci95"][0] = -.001
    decision = choose_coverage(summary, intervals, CONFIG, 5)
    assert decision["model"] == E50
    cv = tmp_path / "cv"
    cv.mkdir()
    summary.to_csv(cv / "summary.csv")
    (cv / "paired_bootstrap.json").write_text(json.dumps(intervals))
    (cv / "run.json").write_text(json.dumps({"config": CONFIG, "exp002_config": {"cv_folds": 5}}))
    decision["sha256"] = {name: file_sha256(cv / name) for name in ("summary.csv", "paired_bootstrap.json", "run.json")}
    (cv / "selection.json").write_text(json.dumps(decision))
    args = SimpleNamespace(run_dir=tmp_path)
    run_coverage_dev(args, None)
    assert not (tmp_path / "dev").exists()
    with (cv / "summary.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="artifact changed"):
        run_coverage_dev(args, None)
