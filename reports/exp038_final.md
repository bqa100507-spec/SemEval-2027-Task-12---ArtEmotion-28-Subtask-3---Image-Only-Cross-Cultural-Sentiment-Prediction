# EXP-038 final closure — cultural-deviation ensemble

**Closed: GO. Selected candidate C2. Best ZIP `B2_C2_cscva3` = round6(0.80 × round6(0.75 × C2 + 0.25 × CS-CVA-3) + 0.20 × C6), labelled `UNCONFIRMED_QWEN_ST3`.** The owner uploaded it to Codabench Development as submission **#1071328** on 2026-11-27 18:09 (UTC+7). The owner reports hosted Cultural Contrast **0.4737**, which matches the local full-Development score (0.473743). For comparison, EXP-037 #1068597 scored 0.4728 and EXP-036 #1065924 scored 0.4703, both owner-reported. The owner adopted `B2_C2_cscva3` as the incumbent. The VLM-free fallback remains the EXP-036 ensemble, then Corrected Ridge. [Runbook](../docs/exp038_kaggle.md) · [Evaluation inference](../docs/exp038_evaluation.md) · [Config](../configs/exp038.json)

Every number below was recomputed on CPU by `scripts/audit_exp038.py` from the archived predictions, without reusing the run's metric JSON. The largest difference from the archived metrics was 7.8e-16, against an audit tolerance of 1e-9, with **no mismatches**. The receipt is `runs/exp038/closure/audit.json`.

## Question and design

Contrast depends only on the image-specific language deviation. Earlier gains came from adding partly independent estimators of that deviation. The hypothesis was that frozen encoders with different pretraining, passed through the unchanged Track 1 full-interaction ridge and the unchanged neighborhood correction, would add Contrast.

- **Arms:** E1 SigLIP2 So400m; E2 MetaCLIP 2 worldwide ViT-H/14; E3 Qwen3-VL-Embedding-2B with a neutral instruction; E4 = E3 plus one shared term on the language-instructed embedding delta.
- **Candidates:** C1 = round6(0.5 Fixed Ridge + 0.5 mean(E1, E2)). C2 adds E3. C3 uses E4 instead of E3. Each candidate then gets the frozen neighborhood correction.
- **Gate:** frozen and pre-registered, run on P2 against the on-platform Corrected Ridge.
- **Development:** reporting only. No coefficient, epoch or candidate was chosen on it.

## Run

Kaggle run 5 finished on 2026-11-27, ending `COMPLETE` at Release with decision GO.

- **Identity:** worker `8b42a10775f0bfa88c4266855f21bce1fb834e94`; overlay manifest `b3303f03…`; base manifest `026720f4…`; one Tesla T4; Python 3.12.10, Torch 2.7.1+cu126, Transformers 4.57.6, qwen-vl-utils 0.0.14, numpy 2.2.6, scipy 1.15.3; Linux 6.18 with OpenBLAS 0.3.29 Haswell kernels and 2 ridge threads.
- **Archive integrity:** all 112 export members match the export manifest. All 10 stage seals match their files and identity; the feature caches seal the run identity plus their arm, spec and chunk size.
- **Budget:** forecast 9,080 s; actual 5,681 s total, leaving 32,119 s unused.
  - E4: forecast 2,138 s, actual 2,099 s (from item start to COMPLETE).
  - CS-CVA: forecast 1,141 s per seed, actual 882 s and 880 s.
  - Other actual times: E1 117 s, E2 73 s, E3 225 s, P2 evaluation 586 s, refit and Development 183 s.
  - Nothing was skipped.

| Artifact (`runs/exp038/closure/`) | Bytes | SHA-256 |
| --- | ---: | --- |
| `exp038_results.zip` | 242,405,211 | `b80e7cda21ee9381b947c2a1707742d4bbf7dff44c864130185b71ccefbe9e09` |
| `exp038_best_submission.zip` | 8,958 | `62093a31da2cd1d864bad7b7818adf082524c09180e75ca0953d5177250ee4e6` |
| `exp038_progress.zip` | 203,502,743 | `3599e2c620c0035d6f3f5bdedbf325404414aa39a038a3ac2a22b2b0ad2840fd` |
| Uploaded overlay `exp038_inputs.zip` | 24,543,206 | `c3adc3b5104bc914f9aa1a003d6c5f41b706a16dc460a0c36f4781b24598370d` |

The best ZIP is byte-identical to `development/submissions/exp038_B2_C2_cscva3_submission.zip` and to `selection.json`. It has 849 rows with unique keys and p in [0.018965, 0.980629].

## R0 replay (cross-platform tolerance)

Validate passed all 33 tolerance checks.

- **Exactness:** 31 checks were `exact=false`. The two exact ones were the bank votes and the bank visual features.
- **Largest difference relative to its threshold:** 25%. This was the P2 Fixed Ridge `clip_large` column, at 5e-6 against a threshold of 2e-5.
- **Per-row differences:** P2 OOF columns differ in 4,895–6,903 of 13,600 rows, by up to 5e-6. Development R0 predictions differ by up to 2e-6.
- **Pooled metrics:** at most 8.3e-8 apart.
- **Heads and bank:** heads within 2.25e-5; bank numerator within 1.37e-5.
- **Diagnostics only:** all five inner hashes and the Corrected Ridge CSV SHA differ, as expected across platforms. These no longer gate.

The gate reference is the on-platform `R0_corrected` recomputed in the same run, and `p2/metrics.json` records this. It differs from the archived Windows CSV in 5,147 rows, by at most 3e-6. All 140 P2 optimizer fits and all 28 Development fits converged, with gradient ≤ 5e-7.

## P2 gate (test-like scope, vs on-platform Corrected Ridge filtered 0.463108)

| Candidate | Filtered Contrast | Δ | 95% CI (20,000 draws, seed 203804) | Fold wins | 5% trim | Full Δ | Filtered LL Δ | Gate |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| C1 (E1+E2, VLM-free) | 0.464804 | +0.001696 | [−0.001472, +0.004809] | 3/5 | +0.001246 | +0.000776 | −0.006188 | FAIL |
| **C2 (E1+E2+E3)** | **0.468648** | **+0.005540** | **[+0.002561, +0.008476]** | **5/5** | +0.003635 | +0.004365 | −0.010657 | **PASS** |
| C3 (E1+E2+E4) | 0.468629 | +0.005521 | [+0.002547, +0.008450] | 5/5 | +0.003439 | +0.004329 | −0.010662 | PASS |

C1 misses the +0.004 gain, the positive lower bound and 4/5 fold wins. C3 passes but loses to C2 under the tie rule: their filtered Contrasts differ by 1.9e-5, under the 0.001 tolerance, and C2 is simpler. Painting wins and losses (filtered) are 638/614 for C1, 673/579 for C2 and 679/573 for C3. The full-P2 Log Loss Δ is −0.006183, −0.010404 and −0.010407.

### Arms (P2 filtered, standalone)

| System | Contrast | Macro LL | Global RMSE | Centered residual r vs Fixed Ridge |
| --- | ---: | ---: | ---: | ---: |
| Fixed Ridge (R0) | 0.460043 | 0.718425 | 0.141391 | — |
| E1 SigLIP2 So400m | 0.451916 | 0.719703 | 0.141728 | 0.982 |
| E2 MetaCLIP 2 worldwide H/14 | 0.451724 | 0.724193 | 0.144672 | 0.981 |
| **E3 Qwen3-VL-Embedding-2B, neutral** | **0.460446** | **0.709892** | **0.137733** | 0.976 |
| E4 language instruction | 0.460419 | 0.709726 | 0.137581 | 0.976 |

- **E3 is the strongest single encoder the project has measured.** On its own it matches the three-encoder Fixed Ridge on Contrast (0.4604 vs 0.4600), with clearly better Log Loss and global RMSE. Almost all of C2's gain over C1 comes from adding E3.
- **E3 and E4 have the lowest residual correlation with Fixed Ridge.** E4 is 0.976002 and E3 is 0.976016, so they are effectively equal. The analysis-side reading that E3 alone is "the lowest of the four" should read "E3 and E4 tie for lowest".
- **No arm falls below the 0.95 threshold of interest**; the range is 0.976–0.982. The gain comes from averaging partly correlated estimators, not from a new kind of signal.
- **E4 is a clean negative.** E4 − E3 is −0.000027, CI [−0.000571, +0.000511], with 652 wins and 600 losses in the filtered scope. Full P2: −0.000056 [−0.000552, +0.000442]. Naming the target language in the embedding instruction adds no image-specific cultural signal.
- **E1 and E2 are each weaker than Fixed Ridge alone.** Together, without a VLM, they do not pass the gate (C1).

## Development (reporting only; 82 full / 79 Evaluation-like eligible paintings)

| System | Full Contrast / LL | Evaluation-like Contrast / LL |
| --- | ---: | ---: |
| On-platform R0_corrected | 0.465462 / 0.740305 | 0.441219 / 0.763903 |
| C1 | 0.470200 / 0.737860 | 0.451667 / 0.761313 |
| C2 | 0.473462 / 0.731046 | 0.450837 / 0.753145 |
| B1_C2_cscva3 | 0.473386 / 0.725035 | 0.454251 / 0.746591 |
| **B2_C2_cscva3 (best ZIP)** | **0.473743 / 0.725075** | **0.456763 / 0.746518** |
| B2_C2_original (single CS-CVA) | 0.474467 / 0.725290 | 0.457343 / 0.746874 |
| EXP-037 original blend (archived) | 0.472818 / 0.728088 | 0.454041 / 0.750275 |
| EXP-036 ensemble (archived) | 0.470339 / 0.729709 | 0.448751 / 0.752288 |

Paired intervals for `B2_C2_cscva3`, against the on-platform recomputed references (20,000 draws):

| Comparison | Full Δ [95% CI] | Evaluation-like Δ [95% CI] |
| --- | --- | --- |
| vs the EXP-037 blend | +0.000925 [−0.007340, +0.009003] | +0.002722 [−0.006111, +0.011377] |
| vs the EXP-036 ensemble | +0.003405 [−0.009045, +0.015587] | +0.008012 [−0.005254, +0.021205] |

All intervals include zero.

All 16 B1/B2 ZIPs and CS-CVA-3 were recomposed from their component ZIPs, and their `predictions.csv` members matched the archive byte for byte. `development_selection` reads "none; Development is reporting only".

### CS-CVA seeds

- **Training:** seeds 203836 and 203936 used the original `configs/exp036.json` recipe with only the seed changed, kept the six-epoch schedule, stopped after epoch 5, and made no Development epoch selection. Each epoch had 197 steps and took a mean of 167.1 s and 166.8 s; Development inference took 5.8 s per seed.
- **Full-Development scores:**

| Model | Full Dev |
| --- | ---: |
| Seed 203836 | 0.430942 |
| Seed 203936 | 0.433540 |
| Original CS-CVA (epoch chosen on Development) | 0.437632 |
| CS-CVA-3 | 0.435131 |

- **Selection optimism (observation only):** the original sits 0.0054 above the mean of the two new seeds and 0.0041 above the better one. That suggests roughly 0.004–0.005 of Development-selection optimism in the original epoch choice.
- **What each blend component adds (observation only).** Stacked on C2, CS-CVA adds about nothing on full Development (B1_C2 0.473386 vs C2 0.473462) and +0.003414 Evaluation-like. C6 then adds +0.000357 full and +0.002512 Evaluation-like (B2 vs B1). The analysis-side figure of "+0.006 Evaluation-like from C6" is the combined CS-CVA plus C6 gain over C2 standalone.
- **No retune:** B2 with the single original CS-CVA scores slightly higher on Development, but the pre-registered rule fixes CS-CVA-3. No coefficient is retuned.

## Eligibility and licensing

- **Qwen eligibility is unconfirmed.** The best ZIP depends on Qwen through E3 and C6, so it is `UNCONFIRMED_QWEN_ST3`; organizer issue #1 is unresolved.
- **MetaCLIP 2 licence.** E2 (MetaCLIP 2 worldwide) is CC-BY-NC-4.0: it must be cited, with no commercial use.
- **No VLM-free improvement.** C1 failed, so the VLM-free fallback stays the EXP-036 ensemble, then Corrected Ridge.

## Limitations

- P2 reuses research paintings and is not an independent test.
- Development has 82 eligible paintings and wide intervals.
- The hosted 0.4737 is a single owner-reported number and adds no new paintings.
- The residual correlations show heavily shared error, so further averaging of similar frozen encoders has a small ceiling.
- The CS-CVA optimism estimate rests on two seeds.

## Technical incident log

Four Kaggle runs stopped before any GPU work, each time on a different packaging or replay issue. Run 5 succeeded.

1. **Run 1: nested archives expanded.**
   - *Symptom:* the overlay's `reference/*.zip` members arrived as directories, failing with "Missing or corrupt file …".
   - *Root cause:* Kaggle expands archives nested inside an uploaded dataset. `.npz` files are ZIP containers too.
   - *Fix (patch 2):* every member with an archive extension or ZIP magic is stored as `.b64` text, decoded into `/kaggle/temp` and checked against its decoded SHA-256. The packager refuses archive-like members, and a test simulates the recursive expansion.
   - *Rule:* pre-upload checklist item 1.
2. **Run 2.** The patch prompts and logs available to this closure do not document a separate run-2 failure, so none is recorded here. Patch 1 was a pre-run change, not a run failure: it made E3/E4 optional so a Qwen failure cannot discard C1 (checklist item 6).
3. **Run 3: truncated pin.**
   - *Symptom:* "Pinned metadata differs … `special_tokens_map.json`".
   - *Root cause:* a hand-copied git-blob pin had 39 characters.
   - *Fix (patch 3):* the value was recomputed from a fresh download. The packager now refuses any malformed pin.
   - *Rule:* checklist item 2.
4. **Run 4: bit-exact replay across platforms.**
   - *Symptom:* "Fold 1 failed exact replay of the Track 1 inner component probabilities".
   - *Root cause:* the iterative ridge solver and platform math libraries make Windows and Linux differ by ≤ 5e-6 in probability and ≤ 1e-7 in metrics.
   - *Fix (patch 4):* replay uses measured tolerances recorded in the config, with receipts. Gates use on-platform references.
   - *Rule:* checklist item 3.
5. **Common root cause:** the implementation was validated only on Windows before handoff. Rule: checklist item 4, Linux preflight from the exact upload ZIP (`scripts/linux_preflight.sh`). Each failure also surfaced one layer at a time (checklist item 5).
6. **Cosmetic:** every Kaggle log line printed twice (launcher plus tee). This is recorded as a known defect for the next launcher template. The EXP-038 runtime is unchanged.

## Possible successors (not designed or started)

- A larger Qwen3-VL-Embedding checkpoint as E3, because E3 alone accounted for C2's gain.
- An ensemble of E3 instructions or resolutions. This is image-level only, since E4's language naming added nothing.
- Drop E4 and the CS-CVA-3 seeds from future recipes. Neither added measurable signal, and both cost about 35 minutes of GPU time.
