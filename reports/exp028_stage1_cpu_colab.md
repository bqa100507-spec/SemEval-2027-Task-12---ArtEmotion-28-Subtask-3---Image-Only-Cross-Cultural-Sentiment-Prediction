# EXP-028 Stage 1 on Colab CPU

TASK-035 supersedes this delivery's runtime instructions with a one-thread
native hotfix and preflight. Use the [hotfix report](task035_colab_lbfgs_hotfix.md)
and its new immutable notebook link in PR #33. The original input receipts below
remain valid.

**EXP028_COLAB = CPU_READY / AWAITING_OWNER_RUN.** The notebook uses the two
existing Drive resources and defaults to the authorized exploratory fold-3
screen. The delivery contains verified input preparation and synthetic workflow
results. Real fold-3 predictions, Contrast, Macro Log Loss, parity and runtime
depend on the completed Colab screen.

1. Open [EXP028_Stage1_CPU_Colab.ipynb](../notebooks/EXP028_Stage1_CPU_Colab.ipynb).
2. In **Runtime → Change runtime type**, choose **CPU**.
3. Choose **Runtime → Run all** and allow the ordinary Drive mount. The default
   paths point to the original handoff and Qwen cache. Set `AUDIT_ONLY=True`
   if you only want to verify inputs.

The notebook shows the nine-model result table after verification. The output
path is `MyDrive/exp028_stage1_cpu/73eabb900b1b/<attempt>/`. It contains
`status.json`, the worker and resource logs, input verification, the frozen
manifest, the Qwen payload audit, fits, evaluation, prediction and painting CSVs.
`manifest_sha256.json` binds the files in `outputs.zip`. The notebook checks
the ZIP after writing it to Drive, optionally requests a browser download and
then requests runtime release. Browser arrival and physical runtime release
depend on Colab and are recorded separately.

## Drive inputs and confirmed coverage

The owner’s complete Colab audit export was retrieved from Drive. Its
SHA-256 is `6010d50a7cf7e4828cddaf16ccdbc296b3d8f0322cc0f606e337971ad53609b9`.
Every file matched its internal manifest. It records **1,335 vision payloads,
13,600 target-language states, 2,670 NPZ receipts, zero invalid entries and zero
remaining forwards**. This supersedes TASK-033’s partial connector listing.
The new runtime independently audits the mounted Qwen payload bytes and their
identity and checks complete vision and language coverage before fitting.

No standalone copies of the three original visual NPZs were established on
Drive. The existing `MyDrive/exp023_handoff.zip` contains the packed visual
banks, the encoder provenance and the artwork hashes required to reconstruct
them **byte for byte**, so there is no separate transfer. The notebook pins the
whole ZIP and its internal manifest. It copies three frozen training and
reference tables, the three visual payload/metadata files, and the 1,335 P2
artworks needed for the existing pixel hash gate. It does not unpack caption
targets, evaluation files or sealed labels. CLIP-L uses the exact original
1,571-row image ordering reconstructed from protocol metadata. The portable
serializer preserves the original NPZ headers across operating systems.

| Frozen cache | Rows in exact NPZ | SHA-256 | Verified P2 keys |
| --- | ---: | --- | ---: |
| CLIP-B | 1,656 | `f0380068c59b2b435b1eaa878293a248b47e0598e8138678039773d6ef849dcb` | 1,335 |
| CLIP-L | 1,571 | `489fdae4ca026c7a6e87d1fb96fe4af4f7c5c98655b6963bb0c2224d33d359fb` | 1,335 |
| SigLIP2-Base | 1,656 | `2d781fce2b5fa57bb53a9be77b8298c9dfe316d25296561a2a77c43f276347eb` | 1,335 |

Both complete visual arrays and each reconstructed file were checked against
the local verified originals. The existing visual loader checks widths, norms,
finite values, key uniqueness and coverage. The worker uses only P2 rows from
these exact caches.

## CPU execution and verification

The notebook pins Git revision `73eabb900b1baef5167affd6f9025e1747326295`,
Python 3.12.10 and the package versions in the CPU lock. The subprocess masks
CUDA and limits BLAS to four threads. The locked environment contains NumPy,
SciPy and scikit-learn. It has no Torch, Transformers or encoder dependencies.
The frozen experiment, data and metric code is byte-checked against TASK-033.
The Stage 1 runner adds only a stricter cache admission option for this
notebook, which requires complete vision payloads before fitting.

The worker verifies the original inputs and refits the E50+L components,
requiring exact frozen six-decimal parity before candidate fitting. It reuses
the existing eight arms, Model C objective, fold, PCA, inner C selection, refit
and paired bootstrap. The wrapper reopens the saved predictions and fit
receipts, verifies held keys and checksums, requires convergence and numerical
parity, and recomputes metrics, painting results and paired intervals. Only
verified outputs are published to Drive. A failure exports the stage, error and
available diagnostic receipts without a partial result table.

The original cache auditor streams one NPZ at a time and retains only decoder
vectors needed by fitting. The wrapper checks effective available memory,
including a Linux memory-group limit if present. Its preflight floor is
4,283,133,880 bytes, derived from the largest raw Model C CSR estimate, the
existing six-times workspace allowance and 1 GiB overhead. The frozen worker
retains its per-fit RAM checks. A low-memory runtime gets a receipt and stops
before fitting. On Linux, resource logs capture real measured wall time and
OS peak child RSS, plus the sampled worker RSS trace. No performance forecast
from a synthetic fit substitutes for these measurements.

## Delivery verification

- Windows CPU suite: **405 passed, 1 skipped**. The notebook schema and workflow
  contract passed. [Machine-readable input and audit receipt](../docs/exp028_colab_cpu_receipt.json).
- The real local handoff restored the P2 assets, and all three reconstructed NPZ
  SHA-256 values matched. The downloaded prior Colab audit receipt independently
  verified the earlier complete coverage report.
- The full synthetic screen and read-back verify all eight arms and matched
  E50+L. Tests exercise invalid sources, missing inputs and dependencies,
  corrupt caches, insufficient RAM, metric replay, parity and convergence,
  rejection of a GPU runtime, ZIP integrity and release ordering.
- This delivery did not execute a real fold-3 fit in Colab. The local session
  had about 2.0 GiB available at input verification, below the preflight floor.
  The reported experiment remains exploratory on previously examined P2 fold 3.
  E50+L remains production.
