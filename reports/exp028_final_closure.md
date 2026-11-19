# EXP-028 Stage 1 final closure

**EXP028 = CLOSED / STAGE1 NO-GO**
**E50+L = UNCHANGED**
**NEXT_DIRECTION = AWAITING_INDEPENDENT_BRAINSTORM**

## Research question and protocol

Could frozen, target-language-conditioned Qwen hidden states complement a frozen visual representation from CLIP-L and SigLIP2-Base? Stage 1 tested whether the correct Qwen-to-painting association improves within-painting Cultural Contrast at matched feature width and whether fusion clears the predefined gains over both the matched E50+L reference and the strongest visual-only control. [Original preregistration](exp028_design_proposal.md) · [Frozen specification](../docs/exp028_design_spec.json) · [Implementation record](exp028_stage1.md)

The screen used P2 fold 3 and eligible votes from the original released training split. The painting-disjoint inner fit and stopping sets contained 961/9,811 and 107/1,068 paintings/pairs, respectively. The refit used their union (1,068 paintings, 10,879 pairs). The held-out fold contained 267 paintings and 2,721 pairs. The eight candidate arms used vote-count-weighted binary L-BFGS logistic regression with language effects and image–language interactions. Each selected C from `{0.1, 0.3, 1.0}` by inner validation BCE, with ties going to lower C, and refit once. PCA used the fitting partition only. The fusion concatenated 256 visual and 256 Qwen components. The shuffle control permuted Qwen states within language and partition while preserving the visual features and head width. The matched E50+L reference used the existing three components at C=0.3 and nested six-decimal blending.

## Run provenance and verification

The owner completed the real Colab CPU run `20261119T063716Z-35957722` on November 19, 2026. The notebook was delivered at `32f1b6e90e183207dc833967b82115eba3eade7f` in [PR #33](https://github.com/bqa100507-spec/SemEval-2027-Task-12---ArtEmotion-28-Subtask-3---Image-Only-Cross-Cultural-Sentiment-Prediction/pull/33). The worker ran source revision `bbbbe3c786fc4ec541cd84b9c30f7e824bf387d1`. The private original is under `MyDrive/exp028_stage1_cpu/bbbbe3c786fc/20261119T063716Z-35957722/`. The closure retained a byte-identical copy of `outputs.zip` in the ignored `runs/task036/` directory. ZIP SHA-256 is `1a13824796d0ea662e9a9a6cc093f0f7b40896c6708883e6c402bc456cbd1abb`.

- The ZIP contained 47 hashed members plus `manifest_sha256.json`. Every member hash passed, as did the 23 files bound by the fold worker receipt and the 16 distinct code/configuration hashes against the immutable Git revision. The workflow, worker, and top-level statuses agree on `FOLD3_SCREEN_COMPLETE / EXPLORATORY`.
- The Colab cache receipt records 1,335 verified artwork files, 1,335 Qwen vision payloads, 13,600 Qwen states keyed by target language, and 2,670 hashed NPZ files covering 1,444,270,098 bytes. It records zero missing or invalid entries and zero remaining forwards. The closure verified the receipt, its hash, its independently derived cache identity `8f1e57257da52d7212b0d8662f0ee46fd82f9dcc64da8f0552498ba40ca5fb43`, and the locally restored visual NPZ hashes. Qwen NPZ bytes were audited by the original worker and were outside the closure ZIP. The closure did not redownload them. [Cache and checkpoint pins](../docs/exp028_feature_inventory.json)
- The handoff ZIP SHA-256 is `2db4a53679940c7eb63c7c8272b91be8ec0618bb6c04a69ebaa15909e20f64e0`, and its manifest SHA-256 is `b5315db3847c29c432e91510a3d8f99c4b1bf13aca8b17282a53059347a20aa9`. The three visual NPZ SHA-256 digests are CLIP-B `f0380068c59b2b435b1eaa878293a248b47e0598e8138678039773d6ef849dcb`, CLIP-L `489fdae4ca026c7a6e87d1fb96fe4af4f7c5c98655b6963bb0c2224d33d359fb`, and SigLIP2-Base `2d781fce2b5fa57bb53a9be77b8298c9dfe316d25296561a2a77c43f276347eb`.
- All 38 recorded fits converged (24 grid fits, eight candidate refits, and six reference-component fits). Recorded objectives agreed with the expanded vote-count form to a maximum absolute error of `3.33e-16`. The recorded C selections and refits agree with the fixed rule. These fit checks use saved receipts. The audit did not rerun the optimizer or PCA.
- Each of the nine prediction CSVs has exactly the same 2,721 unique held-out keys and finite six-decimal probabilities in [0, 1]. Matched E50+L equals all 2,721 historical probabilities exactly. The source historical CSV SHA-256 is `7cc728cd8b50ef828afa5cbaa8ef73c095ec5465664e54c7899d2cbc88039161`. The separate production Development CSV retains SHA-256 `162875b641407efea3bf28d1d147c2d9a3dd3dc2faed8863f9438bee517199a3`.
- From the saved predictions and the same pinned fold-3 votes, the audit reproduced all nine model metrics, all painting CSVs, and all saved paired intervals within absolute `1e-12`. A direct centered-vector Pearson calculation and clipped base-2 cross-entropy independently checked the scorer. Contrast uses 253 of 267 paintings, excluding eight with fewer than three eligible languages and six with constant target fractions. Macro Log Loss equally averages the 21 languages with at least 20 pairs, after averaging losses across pairs per language. This is the repository’s reconstruction of the official scoring rules. [Scoring semantics](../docs/competition.md)

## Colab runtime and hotfix

The real run used x86-64 Linux CPU, CPython 3.12.10, NumPy 2.2.6, SciPy 1.15.3, and scikit-learn 1.6.1 under the pinned lock. All native thread limits were one. SciPy L-BFGS-B, sklearn logistic regression, and the exact regression that previously segfaulted passed in isolated subprocesses. The notebook’s CPU test stage passed 58 tests. This revalidates TASK-035 on the actual Colab path. The precise trigger of the original Colab crash remains unresolved. Worker time was 3,391.01 seconds (56.52 min) with a Linux high-water RSS of 3,288,883,200 bytes (3.06 GiB). The end-to-end notebook reported 3,524.83 seconds before export. [Hotfix investigation](task035_colab_lbfgs_hotfix.md)

## Measured fold-3 results

| Model | Cultural Contrast | Macro Log Loss (bits) |
| --- | ---: | ---: |
| A_raw | 0.435618 | 0.720019 |
| B_raw | 0.344855 | 0.790159 |
| A256 | 0.431200 | 0.713002 |
| B256 | 0.367145 | 0.767040 |
| A512 | 0.434180 | 0.715703 |
| B512 | 0.370652 | 0.765249 |
| **C512** | **0.435013** | **0.711273** |
| C512_shuffled_Q | 0.428455 | 0.712493 |
| Matched E50+L | 0.433536 | 0.713192 |

The deltas and intervals below come from the full-precision archive, which remains unchanged. The paired percentile bootstrap used 10,000 resamples of the 253 eligible paintings with seed 28029. Wins and losses count paintings by the sign of the paired Contrast difference.

| C512 versus | Δ Contrast | 95% paired interval | Wins / losses | Δ Macro Log Loss (bits) |
| --- | ---: | --- | ---: | ---: |
| Matched E50+L | +0.001476 | [−0.005373, +0.008270] | 127 / 126 | −0.001919 |
| A_raw | −0.000605 | [−0.010033, +0.008649] | 128 / 125 | −0.008746 |
| A512 | +0.000833 | [−0.006666, +0.008336] | 122 / 131 | −0.004430 |
| B512 | +0.064361 | [+0.037848, +0.090066] | 166 / 87 | −0.053976 |
| C512_shuffled_Q | +0.006558 | [+0.000778, +0.012448] | 139 / 114 | −0.001221 |

| Preregistered check | Observed | Decision |
| --- | ---: | --- |
| Contrast gain over matched E50+L ≥ +0.003 | +0.001476 | FAIL |
| Contrast gain over strongest visual-only arm ≥ +0.002 | −0.000605 vs A_raw | FAIL |
| Macro Log Loss increase ≤ +0.005 bits vs matched E50+L | −0.001919 | PASS |
| Contrast above the matched-width controls, strongest Qwen-only arm, and shuffled-Q | Above A512, B512 and shuffled-Q | PASS |

## Interpretation and limits

- **Frozen Qwen alone was weaker on this screen.** Even B512, the strongest Qwen-only arm by Contrast, trailed all three visual-only arms. This supports retaining the visual backbone for the tested frozen states and linear readout. The experiment leaves other Qwen layers, pooling, model families and training methods unresolved.
- **The correct association carried useful information within this fusion.** C512 beat its shuffled control and the paired interval excluded zero. That is exploratory evidence specific to the permutation and pipeline. It does not identify a causal mechanism or establish a reliable gain over visual features.
- **The strongest visual-only arm still led.** C512 edged A512 at the same head width and trailed A_raw. The E50+L interval spans zero. Correctly paired Qwen states can contribute information in the fusion without yielding a practically useful improvement over the stronger reference systems.
- **Log Loss and Contrast assess different behavior.** C512 had the lowest observed Macro Log Loss. Better probability forecasts can reduce cross-entropy while leaving cross-language relative variation within a painting similar or worse. Pearson also ignores a common additive shift and positive rescaling within a painting. The lower loss alone cannot isolate calibration from discrimination, and it did not offset either missed Contrast threshold.
- **Fold 3 is historically reused exploratory data.** Paired intervals describe uncertainty conditional on the previously examined paintings and protocol. They establish neither independent confirmation nor performance on other folds or the sealed 236-painting set. No Development or confirmation gold was accessed. No settings were tuned after the results.

The screen completed and the tested proposal is a NO-GO. The gated nonlinear heads, decoder injection and five-fold expansion remain untested and unauthorized. The findings are specific to this proposal and its escalation pathway. Carry forward the matched-width and shuffle results, the stronger raw visual control, and the separation of Contrast from Log Loss into an independent brainstorming session. [Research handoff](exp028_research_handoff.md)

## Read-only reproduction

From an existing checkout, inspect the source ancestry and private export without starting another Colab run. Use the local path to the original export for the hash check.

```powershell
git show --no-patch bbbbe3c786fc4ec541cd84b9c30f7e824bf387d1
Get-FileHash -Algorithm SHA256 runs/task036/outputs.zip
python -m zipfile -t runs/task036/outputs.zip
```

To replay the recorded arithmetic, validate the members against `manifest_sha256.json`. Load the checksum-pinned P2 votes with `st3.exp028_cache.read_protocol(..., labels=True)`, select exactly fold 3, and pass those rows, the nine saved prediction frames and the frozen specification to `st3.exp028_stage1.evaluate`. Compare its result against `fold3/evaluation.json` and the nine painting CSVs. That path uses saved predictions and requires no feature extraction, PCA, fitting, or additional evaluation population. [Frozen fold manifest](../configs/exp028_fold3_manifest.json) · [Archived Colab runbook](exp028_stage1_cpu_colab.md)
