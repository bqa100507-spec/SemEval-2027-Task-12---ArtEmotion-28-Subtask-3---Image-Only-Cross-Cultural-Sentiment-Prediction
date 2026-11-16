import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(2)

from st3 import data
from st3.caption_data import inner_partition, require_isolation
from st3.caption_student import CaptionStudent, concatenate_features, fit_student, predict_student, seed_model, text_loss
from st3.exp023_targets import centered_targets, prepare_partitions
from st3.exp023_data import read_confirmation, reconstruct_history, exact
from st3.exp023_evaluation import aggregate, analyze, paired_interval, scientific_gate
from st3.exp023_runner import Experiment, verify_eligibility
from st3.experiment_workflow import StopRun, publish_model, verify_model, sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    cfg = json.loads((ROOT / "configs/exp023.json").read_text())
    cfg.update(target_dimensions=2, max_epochs=2, patience=1, batch_size=4)
    return cfg


@pytest.fixture
def fixture(config):
    rng = np.random.default_rng(9)
    rows, captions, vectors = [], [], {}
    banks = [{}, {}, {}]
    for p in range(8):
        image = f"p{p}.jpg"
        for width, bank in zip((512, 768, 768), banks):
            vector = rng.normal(size=width).astype(np.float32)
            bank[image] = vector / np.linalg.norm(vector)
        for l in range(3):
            key = f"caption{p}_{l}"
            vector = rng.normal(size=6).astype(np.float32)
            vectors[key] = vector / np.linalg.norm(vector)
            rows.append(dict(painting=f"p{p}", language=f"l{l}", image=image, image_sha256=f"hash{p}",
                             n=5, k=l + 1, split="train", fold=p % 5 + 1))
            for vote in range(5):
                captions.append(dict(painting=f"p{p}", language=f"l{l}", caption_id=key, valid_caption=True,
                                     binary=True, emotion="awe" if vote <= l else "fear", split="train"))
    return pd.DataFrame(rows), pd.DataFrame(captions), vectors, concatenate_features(banks), banks


def test_frozen_config_and_unmodified_historical_source():
    config = json.loads((ROOT / "configs/exp023.json").read_text())
    assert config["conditions"] == ["C0", "C1", "C2"]
    assert config["seeds"] == [2027, 2028, 2029]
    assert config["hidden_width"] == config["target_dimensions"] == 64
    assert (config["max_epochs"], config["batch_size"], config["patience"]) == (150, 256, 15)
    provenance = json.loads((ROOT / "docs/exp023_reused_source.json").read_text())
    for path, expected in provenance["files_sha256"].items():
        assert sha256(ROOT / path) == expected


def test_centering_common_mask_and_weighting(fixture, config):
    pairs, captions, vectors, _, _ = fixture
    result = centered_targets(captions, pairs, vectors, config)
    original, weights, *_ = result["C1"]
    centered = result["C2"][0]
    assert set(original) == set(centered) == set(weights)
    assert len(original) == len(pairs)
    for p in pairs.painting.unique():
        keys = [key for key in original if key[0] == p]
        mean = np.stack([original[k] for k in keys]).mean(0)
        for key in keys:
            expected = original[key] - mean
            expected /= np.linalg.norm(expected)
            np.testing.assert_array_equal(centered[key], expected)
        assert sum(weights[key] for key in keys) == 1.
    assert len(pairs) == result["C1"][4]["all_bce_pairs_retained"]


def test_caption_pca_inner_outer_isolation(fixture, config):
    pairs, captions, vectors, _, _ = fixture
    fitting = pairs[pairs.painting.ne("p7")]
    inner = fitting[fitting.painting.ne("p6")]
    prepared = prepare_partitions(captions, fitting, inner, vectors, config)
    changed = captions.copy()
    changed.loc[changed.painting.isin(["p6", "p7"]), "caption_id"] = "MUST_NOT_BE_READ"
    result = centered_targets(changed, inner, vectors, config)
    for condition in ("C1", "C2"):
        a, b = prepared[condition]["inner"], result[condition]
        np.testing.assert_array_equal(a[2].components_, b[2].components_)
        assert a[4] == b[4]
        assert not set(a[4]["fit_paintings"]) & {"p6", "p7"}


def test_too_few_languages_does_not_remove_bce(fixture, config):
    pairs, captions, vectors, _, _ = fixture
    pairs = pairs[~(pairs.painting.eq("p0") & pairs.language.eq("l2"))]
    result = centered_targets(captions, pairs, vectors, config)
    assert not any(p == "p0" for p, _ in result["C1"][0])
    assert len(pairs[pairs.painting.eq("p0")]) == 2


def test_degenerate_targets_common_mask(fixture, config, monkeypatch):
    pairs, captions, vectors, _, _ = fixture
    from st3 import exp023_targets
    original = exp023_targets.fit_targets
    def exact_degenerate(*args):
        targets, weights, pca, records, metadata = original(*args)
        for key in targets:
            if key[0] == "p0":targets[key] = np.array([1., 0.], dtype=np.float32)
        return targets, weights, pca, records, metadata
    monkeypatch.setattr(exp023_targets, "fit_targets", exact_degenerate)
    result = centered_targets(captions, pairs, vectors, config)
    assert not any(p == "p0" for p, _ in result["C1"][0])
    assert result["C1"][1] == result["C2"][1]


def test_matched_initialization_and_inference_without_supervision(fixture, config):
    pairs, _, _, visual, _ = fixture
    a, _, _ = seed_model(["l0", "l1", "l2"], 2027, config, torch.device("cpu"))
    b, _, _ = seed_model(["l0", "l1", "l2"], 2027, config, torch.device("cpu"))
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    base = pairs[data.KEYS].assign(p_positive=.4)
    q = predict_student(a, pairs[data.INPUT_COLUMNS], visual, base, torch.device("cpu"), 4)
    poisoned = pairs.assign(k=99999, n=-1, caption="forbidden", artist="forbidden", emotion="forbidden")
    np.testing.assert_array_equal(q, predict_student(a, poisoned, visual, base, torch.device("cpu"), 4))
    unseen = pairs.iloc[:1].assign(language="unseen")
    assert predict_student(a, unseen[data.INPUT_COLUMNS], visual, unseen[data.KEYS].assign(p_positive=.32), torch.device("cpu"), 4)[0] == .32


def test_fusion_and_no_backbone_gradient(fixture):
    _, _, _, visual, banks = fixture
    for name, actual in visual.items():
        expected = np.concatenate([bank[name] for bank in banks]) / np.float32(np.sqrt(3))
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (2048,)
        assert not torch.as_tensor(actual).requires_grad


def test_original_auxiliary_normalization():
    predicted = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
    target = torch.tensor([[0., 1.], [1., 0.]])
    weights = torch.tensor([1., .5])
    assert text_loss(predicted, target, weights).item() == 1.
    text_loss(predicted, target, weights).backward()
    assert predicted.grad.abs().sum() > 0


def test_tiny_real_model_fit_seal_reload_and_replay(tmp_path, fixture, config):
    pairs, captions, vectors, visual, _ = fixture
    fitting, held = pairs[pairs.painting.ne("p7")], pairs[pairs.painting.eq("p7")]
    inner, stopping = inner_partition(fitting, config, 1)
    require_isolation(inner, stopping, held)
    prepared = prepare_partitions(captions, fitting, inner, vectors, config)
    model, curves, info = fit_student(fitting, inner, stopping, visual, prepared["C2"], 2027, config, torch.device("cpu"))
    assert curves and info["selected_epoch"] in (1, 2)
    baseline = held[data.KEYS].assign(p_positive=.5)
    raw = predict_student(model, held[data.INPUT_COLUMNS], visual, baseline, torch.device("cpu"), 4)
    identity = {"source": "synthetic_cpu_test", "not_scientific_result": True}
    def writer(stage):
        torch.save({"state_dict": model.state_dict(), "languages": list(model.languages)}, stage / "checkpoint.pt")
    path = tmp_path / "model"
    publish_model(path, identity, writer)
    verify_model(path, identity)
    payload = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
    replay = CaptionStudent(payload["languages"], config)
    replay.load_state_dict(payload["state_dict"], strict=True)
    np.testing.assert_array_equal(raw, predict_student(replay, held[data.INPUT_COLUMNS], visual, baseline, torch.device("cpu"), 4))


def test_round_raw_seeds_before_fixed_blend():
    inputs = pd.DataFrame({"painting": ["p"], "language": ["l"]})
    raw = [np.array([.12345649]), np.array([.12345649]), np.array([.12345749])]
    s, e = aggregate(inputs, raw, inputs.assign(p_positive=.500001))
    assert s.p_positive[0] == np.round(np.mean(raw), 6)
    assert e.p_positive[0] == np.round(.5 * s.p_positive[0] + .5 * .500001, 6)
    with pytest.raises(ValueError):
        aggregate(inputs, raw[:2], inputs.assign(p_positive=.5))


def test_bootstrap_reproducibility():
    a = paired_interval([.01, .02, -.001], 10000, 232027)
    assert a == paired_interval([.01, .02, -.001], 10000, 232027)
    assert a["replicates"] == 10000


def gate_fixture():
    summary = {name: dict(contrast=value, macro_log_loss=.71, fold_mean_macro_log_loss=.71, prediction_variance=.01)
               for name, value in [("R0", .44), ("EC0", .442), ("EC1", .443), ("EC2", .449)]}
    comparisons = {"EC2-" + r: {"ci95": [.001, .02]} for r in ("R0", "EC0", "EC1")}
    rows = [{"fold": f, "model": model, "contrast": values["contrast"]} for f in range(1, 6) for model, values in summary.items()]
    return {"summary": summary, "comparisons": comparisons}, pd.DataFrame(rows)


@pytest.mark.parametrize("failure", [None, "gain_R0", "gain_EC1", "gain_EC0", "ci_R0", "ci_EC1", "loss", "collapse", "models", "inputs", "folds", "worst_fold", "wins_R0", "wins_EC1"])
def test_cv_gate_all_requirements(failure):
    cfg = json.loads((ROOT / "configs/exp023.json").read_text())
    analysis, folds = gate_fixture()
    models, inputs = 45, True
    if failure and failure.startswith("gain_"):
        analysis["summary"][failure[5:]]["contrast"] = .449
    if failure and failure.startswith("ci_"):
        analysis["comparisons"]["EC2-" + failure[3:]]["ci95"][0] = 0.
    if failure == "loss":analysis["summary"]["EC2"]["fold_mean_macro_log_loss"] = .714
    if failure == "collapse":analysis["summary"]["EC2"]["prediction_variance"] = 0.
    if failure == "models":models = 44
    if failure == "inputs":inputs = False
    if failure == "folds":folds = folds[folds.fold.ne(5)]
    if failure == "worst_fold":folds.loc[folds.model.eq("EC2") & folds.fold.eq(1), "contrast"] = .42
    if failure and failure.startswith("wins_"):
        ref = failure[5:]
        folds.loc[folds.model.eq(ref) & folds.fold.le(4), "contrast"] = .449
    result = scientific_gate(analysis, folds, cfg, verified_models=models, inputs_verified=inputs)
    assert result["status"] == ("CV_GO" if failure is None else "CV_NO_GO")


@pytest.mark.parametrize("cv,auth,models", [(False, True, True), (True, False, True), (True, True, False)])
def test_confirmation_no_read_before_all_gates(tmp_path, cv, auth, models):
    with pytest.raises(StopRun, match="sealed"):
        read_confirmation(tmp_path, None, cv_verified=cv, authorized=auth, final_models_verified=models)


def test_no_eligibility_from_checkbox():
    with pytest.raises(StopRun) as error:
        verify_eligibility({"status": "ELIGIBILITY_UNRESOLVED"}, "VERIFIED")
    assert error.value.status == "BLOCKED_ELIGIBILITY"


def test_source_backed_eligibility_validation():
    raw = b"Organizer explicitly permits training-only caption distillation for ST3 with pinned CLIP."
    evidence = dict(status="VERIFIED_ORGANIZER_PERMISSION", covers=["ST3", "original_training_captions_as_auxiliary_targets", "pinned_pretrained_CLIP_L_text"],
                    url="https://github.com/artemotion-28/artemotion-28/issues/999", source_sha256=hashlib.sha256(raw).hexdigest(),
                    exact_permission_excerpt=raw.decode())
    assert verify_eligibility(evidence, "VERIFIED", fetch=lambda url:raw)["status"] == "PASS"
    with pytest.raises(StopRun):verify_eligibility(evidence, "VERIFIED", fetch=lambda url:b"changed")
    evidence["url"] = "https://example.com/fake"
    with pytest.raises(StopRun):verify_eligibility(evidence, "VERIFIED", fetch=lambda url:raw)


def test_reference_exact_mismatch_rejected():
    frame = pd.DataFrame({"painting":["p"], "language":["l"], "fold":[1], "p_positive":[.5]})
    exact(frame, frame)
    with pytest.raises(StopRun):exact(frame, frame.assign(p_positive=.500001))


def test_complete_historical_blend_reconstruction_and_tamper(tmp_path, fixture):
    training = fixture[0]
    history = tmp_path / "history"
    history.mkdir()
    old = []
    for model, value in (("M0", .4), ("M1", .3), ("M2", .5), ("B1", .35), ("B2", .45)):
        old.append(training[[*data.KEYS, "fold"]].assign(model=model, p_positive=value))
    pd.concat(old).to_csv(history / "exp017_predictions.csv", index=False)
    training[[*data.KEYS, "fold"]].assign(p_positive=.4).to_csv(history / "exp022_R0.csv", index=False)
    (history / "independent_verification.json").write_text(json.dumps(dict(status="PASS", checkpoints_replayed=45, baseline_heads_refitted=15, pca_partitions_recomputed=20)))
    for fold in range(1, 6):
        fitting, held = training[training.fold.ne(fold)], training[training.fold.eq(fold)]
        inner, stop = inner_partition(fitting, {"early_stop_seed":17029,"early_stop_fraction":.1}, fold)
        split = {key:sorted(frame.painting.unique()) for key,frame in (("fit",fitting),("held",held),("inner",inner),("stopping",stop))}
        (history/f"fold_{fold}_partition.json").write_text(json.dumps(split))
        (history/f"fold_{fold}_run.json").write_text(json.dumps({"lr_fits":[{"converged":True}]*3}))
        seeds = [held[data.KEYS].assign(model=model,seed=seed,p_positive=value) for model,value in (("M1",.3),("M2",.5)) for seed in (2027,2028,2029)]
        pd.concat(seeds).to_csv(history/f"seeds_fold_{fold}.csv",index=False)
    assert reconstruct_history(tmp_path,training)["status"] == "PASS"
    changed = pd.read_csv(history/"exp017_predictions.csv")
    changed.loc[changed.model.eq("B1"),"p_positive"] += .000001
    changed.to_csv(history/"exp017_predictions.csv",index=False)
    with pytest.raises(StopRun):reconstruct_history(tmp_path,training)


@pytest.mark.parametrize("failure",[None,"loss","gain","ci","models"])
def test_confirmation_gates(failure):
    cfg=json.loads((ROOT/"configs/exp023.json").read_text())
    analysis,folds=gate_fixture()
    models=9
    if failure=="loss":analysis["summary"]["EC2"]["macro_log_loss"] = .714
    if failure=="gain":analysis["summary"]["EC1"]["contrast"] = .449
    if failure=="ci":analysis["comparisons"]["EC2-R0"]["ci95"][0] = -.001
    if failure=="models":models=8
    result=scientific_gate(analysis,folds,cfg,verified_models=models,inputs_verified=True,confirmation=True)
    assert result["status"] == ("CONFIRMATION_GO" if failure is None else "CONFIRMATION_NO_GO")


def test_final_model_identity_change_blocks_promotion(tmp_path, fixture, config):
    # Exercise the production final-model guard without opening any real data.
    engine=object.__new__(Experiment)
    engine.config={**config,"conditions":["C2"],"seeds":[2027]}
    engine.execution_scope="COMPETITION_READY"
    engine.competition_eligible=True
    engine.run_dir=tmp_path
    engine.model_identity=lambda *args:{"synthetic":True}
    path=tmp_path/"models/fold_0_C2_2027"
    publish_model(path,{"synthetic":True},lambda stage:(stage/"checkpoint.pt").write_bytes(b"frozen"))
    (tmp_path/"final_model_identities.json").write_text(json.dumps({"C2_2027":"not-the-confirmed-checkpoint"}))
    with pytest.raises(StopRun,match="changed after confirmation"):engine.final_model()


def test_R0_1335_uses_only_allowed_training_and_resumes(tmp_path, fixture, monkeypatch):
    pairs,_,_,_,banks=fixture
    engine=object.__new__(Experiment)
    from st3.experiment_workflow import Budget
    engine.budget=Budget(4,5,cuda=False)
    engine.operations={"baseline_component_seconds":100}
    engine.execution_scope="COMPETITION_READY"
    engine.competition_eligible=True
    engine.run_dir=tmp_path
    engine.identity={"synthetic":True}
    engine.train=pairs[pairs.painting.ne("p7")]
    engine.banks=banks
    engine.progress=lambda *args,**kwargs:None
    held=pairs[pairs.painting.eq("p7")][data.INPUT_COLUMNS]
    first=engine.final_baseline(held)
    for encoder in ("clip","siglip","large"):
        seal=json.loads((tmp_path/f"R0_1335/{encoder}/seal.json").read_text())
        assert seal["identity"]["kind"]=="R0_1335"
        assert "p7" not in seal["identity"]["fit_paintings"]
        assert len(seal["identity"]["fit_paintings"])==7
    from st3 import exp023_runner
    monkeypatch.setattr(exp023_runner,"fit_frozen_head",lambda *args,**kwargs:pytest.fail("Completed baseline redundantly refit"))
    pd.testing.assert_frame_equal(first,engine.final_baseline(held))
