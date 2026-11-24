import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from st3.data import KEYS
from st3.submission import validate_archive

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("predict_exp036", ROOT / "scripts/predict_exp036.py")
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


def test_fixed_blend_aligns_pairs_and_writes_both_submissions(tmp_path):
    inputs = pd.DataFrame({"painting": ["a", "a", "b"], "language": ["Swahili", "Hindi", "Urdu"],
                           "image": ["a.jpg", "a.jpg", "b.jpg"]})
    q = inputs[KEYS].assign(p_positive=[.666666, 1., 0.])
    p = inputs[KEYS].assign(p_positive=[.234567, 0., .333333])
    hashes = deploy.write_submissions(tmp_path, inputs, p.iloc[::-1], q)
    assert len(hashes) == 4
    echo = validate_archive(tmp_path / "exp036_standalone_submission.zip", inputs)
    pd.testing.assert_frame_equal(echo, p, check_exact=True)
    blend = validate_archive(tmp_path / "exp036_ensemble_submission.zip", inputs)
    np.testing.assert_array_equal(blend.p_positive, [.558641, .75, .083333])
    with pytest.raises(ValueError, match="Pair mismatch"):
        deploy.write_submissions(tmp_path, inputs, p.iloc[:2], q)


def test_fresh_image_launcher_does_not_need_annotations(tmp_path, monkeypatch):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (3, 3), (12, 34, 56)).save(tmp_path / "images/new.png")
    input_path = tmp_path / "st3_test.csv"
    input_path.write_text("painting,language,image\nnew,Swahili,new.png\n")
    exe = tmp_path / "python"
    exe.touch()
    inputs = pd.read_csv(input_path)
    features = {"clip": dict(model="stub",revision="rev",assets_sha256={},width=2,feature="fixture")}
    monkeypatch.setattr(deploy, "verify_bundle", lambda path: None)
    # The fixture uses the actual image/path/bank/ZIP orchestration with stub encoders.
    monkeypatch.setattr(deploy, "load_bundle", lambda path: ({}, {
        "encoders": features, "official_languages": ["Swahili"],
    }))
    bank = Namespace(visual=pd.DataFrame(index=["train"]))
    monkeypatch.setattr(deploy, "load_bank", lambda *args: (bank, set()))
    def worker(name, images, args, spec, scratch):
        assert list(images.values())[0].name == "new.png"
        return {h: np.ones(2) for h in images}, {"identity": spec}
    monkeypatch.setattr(deploy, "run_feature_worker", worker)
    monkeypatch.setattr(deploy, "ridge_with_features", lambda *a: inputs[KEYS].assign(p_positive=[.8]))
    def infer(command, check):
        assert set(command[command.index("infer")+1::2]) == {
            "--bundle", "--pairs", "--images", "--destination", "--microbatch",
        }
        csv = Path(command[command.index("--destination")+1])
        inputs[KEYS].assign(p_positive=[.2]).to_csv(csv, index=False)
    monkeypatch.setattr(deploy.subprocess, "run", infer)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    out = tmp_path / "out"
    args = Namespace(bundle=bundle, input=input_path, data_dir=tmp_path,
                     output=out, ridge_only=False, cscva_python=exe, model=None)
    deploy.run(args)
    assert validate_archive(out / "exp036_ensemble_submission.zip", inputs).p_positive.tolist() == [.65]
    assert deploy.read(out / "run.json")["cs_cva_weight"] == .25
    args.ridge_only = True
    args.cscva_python = None
    args.output = tmp_path / "fallback"
    deploy.run(args)
    assert validate_archive(args.output / "corrected_ridge_submission.zip", inputs).p_positive.tolist() == [.8]
    input_path.write_text("painting,language,image,k\nnew,Swahili,new.png,3\n")
    args.output = tmp_path / "forbidden"
    with pytest.raises(ValueError, match="Expected CSV columns"):
        deploy.run(args)
