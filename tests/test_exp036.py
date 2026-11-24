import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from st3.exp036 import Candidate, centered_targets, choose_epoch, finish, loss_parts
from st3.exp036_io import export, restore
from st3.submission import validate_archive

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/exp036.json").read_text())


def worker(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny():
    cfg = CFG | {"hidden_size": 16, "layers": 2, "adapter_layers": 1, "rank": 2, "caption_width": 4}
    siglip = pytest.importorskip("transformers.models.siglip.modeling_siglip")
    native = siglip.SiglipVisionModel(siglip.SiglipVisionConfig(hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=2, image_size=8, patch_size=4))
    return native, cfg


def test_multiplicity_and_centering():
    # Their raw means are [1,0], [0,1], [0,1] and [1,0].
    counts = np.array([3, 1, 1, 1, 1])
    totals = np.array([[3,0],[0,1],[0,1],[1,0],[1,1]],dtype=np.float32)
    values, mask, summary = centered_targets(totals, counts, np.array([0,0,1,1,2]),
                                            np.array([0,1,0,1,0]), 1e-6)
    assert mask.tolist() == [True, True, True, True, False]
    np.testing.assert_allclose(values[0], -values[1])
    np.testing.assert_allclose(np.linalg.norm(values[mask], axis=1), 1., atol=1e-6)
    assert summary["caption_pairs"] == 5 and summary["target_pairs"] == 4
    assert summary["language_mean_norm"] != summary["painting_mean_norm"]


def test_real_siglip_module_layout_initial_parity_and_gradient():
    native, cfg = tiny()
    pixels = torch.randn(2,3,8,8)
    expected = native.vision_model(pixels).pooler_output
    model = Candidate(native.vision_model, 3, cfg)
    assert len(model.targets) == 4 and all(name.startswith("encoder.layers.1.") for name in model.targets)
    torch.testing.assert_close(model.features(pixels), F.normalize(expected, dim=-1), rtol=2e-6, atol=2e-6)
    group, lang = torch.tensor([0,0,1]), torch.tensor([0,1,2])
    y, n = torch.tensor([4.,5.,0.]), torch.tensor([5.,7.,5.])
    targets = F.normalize(torch.randn(3,16),dim=1)
    bce, cosine = loss_parts(model, pixels,(group,lang,y,n,targets,n),n.sum(),n.sum(),1e-6)
    assert bce.item() == pytest.approx(np.log(2))
    (bce+.1*cosine).backward()
    assert any(p.grad.abs().sum() > 0 for name,p in model.named_parameters() if name.endswith(".right"))
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    assert model.predict(pixels, torch.tensor([0]), torch.tensor([-1])).isfinite().all()


def test_grouped_microbatch_objective_matches_unchunked():
    torch.manual_seed(12)
    native, cfg = tiny()
    model = Candidate(native.vision_model, 2, cfg)
    pixels = torch.randn(2,3,8,8)
    target = F.normalize(torch.randn(4,16),dim=1)
    languages = torch.tensor([0,1,0,1])
    counts, pos = torch.tensor([5.,5.,5.,8.]), torch.tensor([1.,4.,0.,8.])
    mask = torch.tensor([5.,5.,0.,8.])
    trainable = [p for p in model.parameters() if p.requires_grad]
    def loss(pixel, ix, group):
        bce, cap = loss_parts(model,pixel,(group,languages[ix],pos[ix],counts[ix],target[ix],mask[ix]),
                              counts.sum(),mask.sum(),1e-6)
        return bce+cfg["caption_lambda"]*cap
    one = loss(pixels,slice(None),torch.tensor([0,0,1,1]))
    split = sum(loss(pixels[i:i+1],slice(i*2,i*2+2),torch.tensor([0,0])) for i in range(2))
    torch.testing.assert_close(one,split)
    for a,b in zip(torch.autograd.grad(one,trainable),torch.autograd.grad(split,trainable),strict=True):
        torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-6)


def test_resume_optim_scheduler_rng_and_archive(tmp_path):
    run = worker("run_exp036")
    train = tmp_path / "train"
    train.mkdir()
    model = torch.nn.Linear(3,2)
    def setup(model):
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        return opt, torch.optim.lr_scheduler.StepLR(opt,1), torch.amp.GradScaler("cpu",enabled=False)
    def step(model, opt, schedule):
        opt.zero_grad()
        model(torch.randn(3)).square().sum().backward()
        opt.step()
        schedule.step()
    opt, sch, scale = setup(model)
    step(model,opt,sch)
    device = torch.device("cpu")
    identity = {"source":"fixture"}
    run.checkpoint(train,identity,model,opt,sch,scale,{"epoch":1,"step":1,"history":[]},device)
    export(tmp_path/"progress.zip",train,run.checkpoint_files(train),"input")
    decoded = tmp_path/"resume"
    decoded.mkdir()
    restore(tmp_path/"progress.zip",decoded,"input")
    step(model,opt,sch)
    restored = torch.nn.Linear(3,2)
    nopt, nsch, nscale = setup(restored)
    state = run.resume_state(decoded,identity)
    run.apply_state(restored,state["model"])
    nopt.load_state_dict(state["optimizer"])
    nsch.load_state_dict(state["scheduler"])
    nscale.load_state_dict(state["scaler"])
    run.restore_rng(state["rng"],device)
    step(restored,nopt,nsch)
    for expected, actual in zip(model.parameters(),restored.parameters(),strict=True):
        torch.testing.assert_close(expected,actual,atol=0,rtol=0)
    with pytest.raises(ValueError, match="identity"):
        run.resume_state(decoded,{"source":"wrong"})


def fixture_predictions():
    dev = pd.DataFrame([(str(i), lang, f"{i}.jpg") for i in range(21) for lang in ("A","B","C")],
                       columns=["painting","language","image"])
    gold = dev[["painting","language"]].assign(k=[1,2,4]*21,n=5)
    incumbent = dev[["painting","language"]].assign(p_positive=[.6,.4,.7]*21)
    weak = dev[["painting","language"]].assign(p_positive=[.8,.6,.2]*21)
    return dev,gold,incumbent,weak


def test_dev_selection_and_independent_submission_even_when_standalone_loses(tmp_path):
    dev,gold,incumbent,weak = fixture_predictions()
    result = finish(dev,gold,weak.iloc[::-1],incumbent,tmp_path,CFG)
    assert result["ensemble"]["alpha"] == 0
    assert result["standalone"]["cultural_contrast"] < result["reference"]["cultural_contrast"]
    for kind, values in (("standalone",weak),("ensemble",incumbent)):
        file = tmp_path / f"exp036_{kind}_submission.zip"
        pd.testing.assert_frame_equal(validate_archive(file,dev),values)
    with pytest.raises(ValueError,match="Duplicate"):
        finish(dev,gold,pd.concat([weak,weak.iloc[:1]]),incumbent,tmp_path,CFG)
    assert choose_epoch([{"epoch":2,"metrics":result["reference"]},
                         {"epoch":1,"metrics":result["reference"]}])["epoch"] == 1


def test_notebook_launch_validates_inputs_before_invoking_worker(tmp_path,monkeypatch):
    module = worker("package_exp036")
    stage = tmp_path / "input" / "test" / "exp036_inputs"
    (stage/"runtime/scripts").mkdir(parents=True)
    (tmp_path/"working").mkdir()
    p = stage / "runtime/scripts/exp036_kaggle.py"
    p.write_text("# worker")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = stage / "manifest.json"
    manifest.write_text(json.dumps({"files":{"runtime/scripts/exp036_kaggle.py":sha(p)}}))
    nb = module.notebook(sha(manifest),"unused")
    cells = {c["id"]:"".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"}
    for name,code in cells.items():
        compile(code,name,"exec")
    calls = []
    monkeypatch.setattr(sys.modules["subprocess"],"run",lambda args,**kw: calls.append(args))
    variables = {}
    exec(cells["settings"],variables)
    variables["AUTHORIZE_RUN"] = True
    launch = cells["launch"].replace("/kaggle/",tmp_path.as_posix()+"/")
    exec(launch,variables)
    assert len(calls) == 1 and calls[0][2] == str(p)
    p.write_text("# corrupted")
    exec(launch,variables)
    assert len(calls) == 1
    assert "Missing or corrupt" in (tmp_path/"working/exp036_bootstrap_error.json").read_text()
