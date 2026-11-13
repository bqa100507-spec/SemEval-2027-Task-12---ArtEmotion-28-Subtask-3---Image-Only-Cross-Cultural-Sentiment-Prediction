"""Regenerate the two thin notebooks with an immutable helper commit and digest."""

import argparse
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE_REV='41ca0d4061f129c1821009e6d00614dede08708f'
HELPER_REV='b18991733a6ec60493e35985fcaff945a164d9f4'
REPO='bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction'

BOOTSTRAP=r'''
import hashlib, importlib.util, json, re, sys, time, traceback, urllib.request, zipfile
from pathlib import Path

RUNTIME = None
OFFLINE_KIT = None
OFFLINE_SHA = None
RESTORE_PATH = ""
RESTORE_SHA = ""
BOOTSTRAP_OK = False

def published_sha(path, configured):
    # Read a previously published checksum; never derive a replacement from the ZIP.
    checksum_file = path.with_suffix(".sha256")
    expected = configured or (checksum_file.read_text().split()[0] if checksum_file.is_file() else "")
    if not re.fullmatch("[0-9a-f]{64}", expected):
        raise RuntimeError(f"Missing published checksum for {path.name}. Attach its .sha256 file or set its SHA field from the producer's receipt.")
    return expected

def choose_artifact(explicit, candidates, description):
    if explicit:
        value = Path(explicit)
        if not value.is_file(): raise RuntimeError(f"{description} not found: {value}")
        return value
    values = sorted(candidates)
    if len(values) != 1: raise RuntimeError(f"Found {len(values)} {description} files. Attach exactly one or set its explicit path in configuration.")
    return values[0]

if MOUNT_OK:
    try:
        log_root = Path(PERSISTENT_ROOT) / "setup_logs"
        log_root.mkdir(parents=True, exist_ok=True)
        tools = Path(LOCAL_ROOT) / "tools"
        tools.mkdir(parents=True, exist_ok=True)
        search = Path("/kaggle/input") if PLATFORM == "Kaggle" else Path("/content/drive/MyDrive")
        candidates = list(search.rglob("exp022_runtime.zip")) if PLATFORM == "Kaggle" else list(search.glob("exp022_runtime.zip"))
        if NETWORK_MODE not in ("auto", "online", "offline"):
            raise RuntimeError("Network mode must be auto, online or offline.")
        offline = NETWORK_MODE == "offline" or (NETWORK_MODE == "auto" and (OFFLINE_KIT_PATH or candidates))
        if offline:
            OFFLINE_KIT = choose_artifact(OFFLINE_KIT_PATH, candidates, "offline runtime ZIP")
            OFFLINE_SHA = published_sha(OFFLINE_KIT, OFFLINE_KIT_SHA256)
            digest = hashlib.sha256()
            with OFFLINE_KIT.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
            if digest.hexdigest() != OFFLINE_SHA:
                raise RuntimeError("Offline kit SHA mismatch. Reattach the original kit and checksum; do not replace the expected hash.")
            with zipfile.ZipFile(OFFLINE_KIT) as archive:
                if archive.namelist().count("exp022_notebook.py") != 1:
                    raise RuntimeError("Offline kit needs exactly one pinned exp022_notebook.py.")
                helper_bytes = archive.read("exp022_notebook.py")
            print("Offline setup selected: no GitHub or pip network access will be requested.")
        else:
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{HELPER_REVISION}/scripts/exp022_notebook.py"
            try:
                with urllib.request.urlopen(url, timeout=30) as response: helper_bytes = response.read()
            except Exception as error:
                raise RuntimeError("Internet/helper download unavailable. Enable Notebook Internet, or attach exp022_runtime.zip plus its published SHA from a prior online benchmark. The 705 MB data ZIP is NOT a complete offline environment. Details: " + str(error)) from None
            print("Online helper download succeeded. GitHub/PyPI/PyTorch access is also required for setup.")
        if hashlib.sha256(helper_bytes).hexdigest() != HELPER_SHA256:
            raise RuntimeError("Helper SHA mismatch. Keep this notebook's pins; use the matching published helper/offline kit.")
        helper_path = tools / ("exp022_notebook-" + HELPER_SHA256[:12] + ".py")
        helper_path.write_bytes(helper_bytes)
        spec = importlib.util.spec_from_file_location("exp022_cloud_workflow", helper_path)
        RUNTIME = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(RUNTIME)
        if RUNTIME.SOURCE_REV != TRAINING_REVISION:
            raise RuntimeError("Helper training revision differs from the frozen experiment.")
        if RESTORE_CHECKPOINTS:
            candidates = search.rglob("exp022-checkpoints-*.zip") if PLATFORM == "Kaggle" else search.glob("exp022-checkpoints-*.zip")
            RESTORE_PATH = str(choose_artifact(RESTORE_BUNDLE_PATH, candidates, "checkpoint bundle"))
            RESTORE_SHA = published_sha(Path(RESTORE_PATH), RESTORE_BUNDLE_SHA256)
        if RUN_LONG_TRAINING:
            raise RuntimeError("Run all must start with RUN_LONG_TRAINING=False. Use the post-benchmark stage controls to opt in.")
        BOOTSTRAP_OK = True
    except Exception as error:
        print("STOP — setup failed:", str(error))
        if "log_root" in globals() and log_root.exists():
            failure_log = log_root / ("bootstrap-error-" + str(time.time_ns()) + ".log")
            failure_log.write_text(traceback.format_exc(), encoding="utf-8")
            print("Full diagnostic log:", failure_log)
else:
    print("STOP — Drive is not mounted. Authorize the mount, then Run all again.")
'''

BENCHMARK=r'''
from IPython.display import display, Markdown, FileLink

SESSION = None
BENCHMARK_OK = False
if BOOTSTRAP_OK:
    try:
        paths = RUNTIME.paths_for(PLATFORM, input_path=INPUT_ZIP_PATH,
                                  local_root=LOCAL_ROOT, persistent_root=PERSISTENT_ROOT)
        SESSION = RUNTIME.NotebookRun(PLATFORM, paths, offline_kit=OFFLINE_KIT)
        SESSION.setup(offline_sha=OFFLINE_SHA, restore_path=RESTORE_PATH, restore_sha=RESTORE_SHA)
        # The ONLY automatic experiment stages are Preflight (600s) and Pilot (900s).
        SESSION.benchmark()
        BENCHMARK_OK = True
        display(Markdown(RUNTIME.report_markdown(SESSION.summary())))
        if SESSION.last_export:
            display(FileLink(SESSION.last_export["path"]))
            print("Export SHA-256:", SESSION.last_export["sha256"])
    except Exception as error:
        if SESSION is not None:
            SESSION.failure(error)
            display(Markdown(RUNTIME.report_markdown(SESSION.summary())))
        else:
            print("STOP — preparation failed:", str(error))
else:
    print("Benchmark not started. Fix the first setup error above, then Run all again.")
'''

CONTROLS='''
# Register buttons only. Run all never clicks a button or launches these stages.
if BENCHMARK_OK:
    RUNTIME.show_controls(SESSION, globals())
else:
    print("Long-stage controls are disabled until setup, Preflight and Pilot pass.")
'''


def notebook(platform, helper_revision, helper_sha):
    cells=[]
    def add(kind, source):
        value={'cell_type':kind,'id':f'{platform.lower()}-{len(cells):02}', 'metadata':{},
               'source':textwrap.dedent(source).strip().splitlines(keepends=True)}
        if kind=='code':value.update(execution_count=None,outputs=[])
        cells.append(value)
    if platform=='Colab':
        steps='''1. Upload **exp022_colab_inputs.zip** to the root of **MyDrive**.
2. Select **Runtime → Change runtime type → GPU**.
3. Choose **Runtime → Run all** and authorize the Drive mount.
4. Read the benchmark summary. Default Run all does **not** launch screening.

Only change **INPUT_ZIP_PATH** below if your ZIP is elsewhere. First setup downloads the exact Python and pinned dependencies; no notebook-kernel restart is needed.'''
        input_path='/content/drive/MyDrive/exp022_colab_inputs.zip';local='/content/exp022';persistent='/content/drive/MyDrive/exp022_runs'
    else:
        steps='''1. Create a **private Kaggle Dataset** containing **exp022_colab_inputs.zip**.
2. Import this notebook into a **private** Kaggle Notebook and **Add Input** with that Dataset.
3. Enable a **GPU** accelerator and **Internet**, then **Run all**.
4. Read the benchmark summary. The helper uses **one GPU**, including on dual-T4 runtimes.

The ZIP is detected automatically. Missing/duplicate ZIPs stop with an actionable message. For Internet-disabled sessions, also attach the separately prepared **exp022_runtime.zip** and its **.sha256** receipt; the data ZIP alone is insufficient. See the offline instructions below.'''
        input_path='';local='/kaggle/temp/exp022';persistent='/kaggle/working/exp022_runs'
    add('markdown',f'''# EXP-022 — {platform}

{steps}

This is an execution interface for the existing four-condition experiment, not a new model experiment. Frozen source: `{SOURCE_REV}` (same training implementation as `f0d2ac6`). All architecture, losses, folds, seeds and scientific gates stay unchanged. **Cloud GPU compatibility is established only when this runtime's actual Preflight/Pilot pass.**''')
    add('markdown','''## Configuration

Defaults are ready for an online benchmark. Advanced fields are optional: offline kit, explicit checkpoint restore, and output directory. Expected data/helper hashes are fixed below; never replace them to bypass a failure. `auto` uses an attached offline kit when present; otherwise it requires Internet.''')
    config=f'''
PLATFORM = {platform!r}
INPUT_ZIP_PATH = {input_path!r}  # @param {{type:"string"}}
NETWORK_MODE = "auto"  # @param ["auto", "online", "offline"]
OFFLINE_KIT_PATH = ""  # @param {{type:"string"}}
OFFLINE_KIT_SHA256 = ""  # optional; otherwise use its published .sha256 file
RESTORE_CHECKPOINTS = False  # @param {{type:"boolean"}}
RESTORE_BUNDLE_PATH = ""  # optional; auto-detect one attached checkpoint ZIP
RESTORE_BUNDLE_SHA256 = ""  # optional; otherwise use its published .sha256 file
LOCAL_ROOT = {local!r}
PERSISTENT_ROOT = {persistent!r}  # @param {{type:"string"}}

# Safety default: only the explicit post-benchmark button can opt into long work.
RUN_LONG_TRAINING = False
TRAINING_REVISION = {SOURCE_REV!r}
REPOSITORY = {REPO!r}
HELPER_REVISION = {helper_revision!r}
HELPER_SHA256 = {helper_sha!r}
MOUNT_OK = True
'''
    add('code',config)
    if platform=='Colab':
        add('markdown','''## Mount Drive

Authorize your own Drive when prompted. Images, weights, Python and the training checkout are extracted/installed locally under `/content/exp022`; only run outputs and logs use Drive.''')
        add('code','''
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as error:
    MOUNT_OK = False
    print("STOP — Drive mount failed. Open this notebook in Google Colab, authorize Drive, and Run all again. Details:", str(error))
''')
    add('markdown','''## Verify workflow tools

Online mode loads one checksum-pinned helper from its immutable Git commit. Offline mode verifies the separately published runtime ZIP checksum and then the same helper checksum before executing it. Training is cloned from the separate frozen revision, never branch HEAD.''')
    add('code',BOOTSTRAP)
    add('markdown','''## Prepare and benchmark

The helper validates the original ZIP, bundle and input-manifest hashes, every extracted asset, the source checkout and exact Python 3.12.10 / Torch 2.7.1+cu126 / Transformers 4.49.0 environment. It refuses missing CUDA or incompatible dependencies. The host notebook's Python/Torch do not replace the isolated environment.

It runs **Preflight → Pilot** only. Full diagnostic logs are saved separately; the first actionable failure is shown here. A parity failure must be investigated without relaxing its tolerance or changing scientific settings.''')
    add('code',BENCHMARK)
    add('markdown','''## Optional stages — explicit owner action only

After reviewing the summary, select a stage, review its runtime/VRAM and budget, check authorization, then click **Run selected stage**. Stages are `Setup / Benchmark`, `MiniScreen`, `MiniScreenVerify`, `Screen`, `ScreenVerify`. The helper and original runner both enforce prerequisites; a failed replay invalidates the previous decision. The four-GPU-hour Screen forecast gate remains in effect.

Run all only creates these controls; it does not press a button. Authorization resets after use. Full five-fold CV and its final verification are not exposed here. Benchmark losses/gradients do not establish held-out quality or justify promotion.''')
    add('code',CONTROLS)
    persistence='''Drive keeps published complete checkpoints, stage manifests and logs in the displayed identity-specific run directory. Colab can still terminate a process or disconnect Drive: only models already fully published are guaranteed reusable. The current partial model restarts. Repeat Run all to reopen the same compatible run; reused benchmark measurements are explicitly labeled.''' if platform=='Colab' else '''`/kaggle/working` is temporary until you preserve outputs. The helper exports versioned checkpoint/log ZIPs plus `.sha256` receipts after stages, including interruption handling. **Download the latest ZIP and receipt before ending a session**, or preserve them as private versioned Notebook output / a private Dataset and verify they are present. A plain session directory is not a persistence guarantee.

Kaggle **Save & Run All** re-executes the safe benchmark; it does not reproduce interactive button clicks. Export the interactive checkpoints first. On a new session, attach the private checkpoint ZIP and receipt, set `RESTORE_CHECKPOINTS=True`, and Run all. Only complete published models restore; no partial-model recovery is promised. Dependencies and assets use `/kaggle/temp`, so they are not accidentally saved as large Notebook outputs.'''
    add('markdown',f'''## Persistence, switching platforms and offline Kaggle

{persistence}

**Switching platforms:** use the same data ZIP, notebook pins and scientific configuration. Transfer artifacts with **Export completed artifacts**, and attach that ZIP plus its `.sha256` receipt. Restore requires exact scientific and GPU/software/driver/dependency-lock identities. A different identity starts a separate run; it cannot reuse old fits silently. Keep incompatible checkpoints archived and benchmark/retrain on the new runtime.

**Offline kit:** after a successful online Linux benchmark, use **Build offline Kaggle kit** (explicit authorization; several GB of network/storage, no model training). It creates `exp022_runtime.zip` plus `.sha256`: the frozen Git source bundle, relocatable Python 3.12.10, pinned helper and exact dependency wheels/lock. Keep it private and attach it alongside the original data ZIP on Kaggle. `NETWORK_MODE=auto` detects it; offline setup uses `--no-index` and never assumes the 705 MB data ZIP contains dependencies. This packaging mechanism has synthetic validation, not a completed cloud GPU compatibility certification.

**Troubleshooting:** missing ZIP → upload/attach it or set its path; duplicate ZIP → detach duplicates or choose an explicit path; missing CUDA → enable a GPU; blocked Internet → enable it or attach the complete offline kit/receipt; wrong hash → replace the artifact, not the expected checksum; parity failure → inspect saved logs and preserve the failed run; incompatible restore → keep the archive and use a separate runtime/run directory.

No locked confirmation data, public Development gold, alternative pretrained checkpoint, Codabench upload or production E50+L modification is permitted. [Experiment report](https://github.com/{REPO}/blob/{SOURCE_REV}/reports/exp022_visual_lora_vote_distribution.md) · [Colab runtime/storage guidance](https://research.google.com/colaboratory/faq.html) · [Kaggle outputs guidance](https://www.kaggle.com/docs/notebooks) · [Pinned Python installation method](https://docs.astral.sh/uv/guides/install-python/)''')
    metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'},
              'exp022':{'training_revision':SOURCE_REV,'helper_revision':helper_revision,'helper_sha256':helper_sha,'default_stages':['Preflight','Pilot']}}
    if platform=='Colab':metadata.update(colab={'name':'EXP022_Colab.ipynb','provenance':[],'toc_visible':True},accelerator='GPU')
    else:metadata['kaggle']={'isInternetEnabled':True,'isGpuEnabled':True,'isTpuEnabled':False,'language':'python','sourceType':'notebook'}
    return {'cells':cells,'metadata':metadata,'nbformat':4,'nbformat_minor':5}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--helper-revision',default=HELPER_REV)
    args=parser.parse_args()
    content=subprocess.check_output(['git','show',args.helper_revision+':scripts/exp022_notebook.py'],cwd=ROOT)
    helper_sha=hashlib.sha256(content).hexdigest()
    for name in ('Colab','Kaggle'):
        path=ROOT/'notebooks'/f'EXP022_{name}.ipynb';path.parent.mkdir(exist_ok=True)
        path.write_bytes((json.dumps(notebook(name,args.helper_revision,helper_sha),indent=1,ensure_ascii=False)+'\n').encode('utf-8'))
        print(path,helper_sha)


if __name__=='__main__':main()
